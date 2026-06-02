"""End-to-end HARVE pipeline for one RM:

    1. Extract per-subcategory directions (writes runs/harve/{rm}_directions.pt)
    2. Cache test-set hidden states (writes runs/harve/{rm}_test_cache.pt)
    3. α-sweep — produces RewardHackBench scores at every α and selects α*
    4. Apply α* and save the final edited w_r as runs/harve/{rm}_w_r_star.pt

The script is idempotent: each step is skipped if its output already exists.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.harve.edit_reward_head import ablate_w_r
from src.harve.extract_directions import extract_all_directions
from src.utils import (
    apply_chat_template_pair,
    iter_pairs_by_subcategory,
    load_benchmark,
    load_rm_for_caching,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Step 2: cache test-set hidden states
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_test_hiddens(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                      *, output_dir: Path) -> Path:
    out_path = output_dir / f"{rm_key}_test_cache.pt"
    if out_path.exists():
        print(f"  [cache] {out_path.name} exists — skipping")
        return out_path

    spec = rms_cfg["models"][rm_key]
    model, tok = load_rm_for_caching(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )

    test_pairs = load_benchmark("test")
    print(f"  Caching test hiddens for {rm_key}: {len(test_pairs)} pairs")
    cache: dict[tuple[str, int, str], torch.Tensor] = {}
    for i, r in enumerate(tqdm(test_pairs, desc="    caching")):
        for role, resp in [("gold", r["gold_response"]),
                           ("hacked", r["hacked_response"])]:
            enc = apply_chat_template_pair(tok, r["question"], resp,
                                           max_length=harve_cfg.get("caching", {}).get("max_length", 1024))
            ids = enc["input_ids"].to(model.device)
            mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            pos = -1 if getattr(tok, "padding_side", "right") == "left" else mask[0].sum().item() - 1
            cache[(r["category"], i, role)] = out.hidden_states[-1][0, pos, :].float().cpu()

    torch.save({
        "rm_key": rm_key,
        "cache": cache,
        "n_pairs": len(test_pairs),
    }, out_path)
    print(f"  Saved → {out_path}")
    del model
    torch.cuda.empty_cache()
    return out_path


# --------------------------------------------------------------------------- #
# Step 3: α-sweep
# --------------------------------------------------------------------------- #
def alpha_sweep(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                *, output_dir: Path) -> Path:
    out_path = output_dir / f"{rm_key}_alpha_sweep.json"
    if out_path.exists():
        print(f"  [sweep] {out_path.name} exists — skipping")
        return out_path

    dirs_path = output_dir / f"{rm_key}_directions.pt"
    cache_path = output_dir / f"{rm_key}_test_cache.pt"
    assert dirs_path.exists() and cache_path.exists()

    dirs_bundle = torch.load(dirs_path, map_location="cpu", weights_only=False)
    cache_bundle = torch.load(cache_path, map_location="cpu", weights_only=False)

    w_r = dirs_bundle["w_r"].float()
    target_cats = dirs_bundle["target_categories"]
    v_dict = dirs_bundle["v_dict"]
    V = torch.stack([v_dict[c] for c in target_cats if c in v_dict]).float()
    cache = cache_bundle["cache"]

    test_pairs = load_benchmark("test")
    cats = sorted({r["category"] for r in test_pairs})

    def score_at_alpha(alpha: float) -> dict:
        w_a = ablate_w_r(w_r, V, alpha,
                          svd_threshold=harve_cfg["ablation"]["svd_threshold"])
        per_cat = {}
        for cat in cats:
            n_correct = 0
            n_total = 0
            for i, r in enumerate(test_pairs):
                if r["category"] != cat:
                    continue
                h_g = cache[(cat, i, "gold")]
                h_h = cache[(cat, i, "hacked")]
                s_g = float(h_g @ w_a)
                s_h = float(h_h @ w_a)
                n_total += 1
                if s_g > s_h:
                    n_correct += 1
            per_cat[cat] = {"n": n_total, "correct": n_correct,
                            "accuracy": 100.0 * n_correct / n_total if n_total else float("nan")}
        # Overall = sample-weighted micro across every pair
        total_n = sum(per_cat[c]["n"] for c in per_cat)
        total_c = sum(per_cat[c]["correct"] for c in per_cat)
        per_cat["_overall"] = {
            "n": total_n, "correct": total_c,
            "accuracy": 100.0 * total_c / total_n if total_n else float("nan")}
        return per_cat

    sweep = []
    for a in harve_cfg["ablation"]["alphas"]:
        sweep.append({"alpha": a, "per_cat": score_at_alpha(a)})
        print(f"    α={a:>5}  overall={sweep[-1]['per_cat']['_overall']['accuracy']:.2f}%")

    out_path.write_text(json.dumps({
        "rm_key": rm_key,
        "alphas": harve_cfg["ablation"]["alphas"],
        "sweep": sweep,
    }, indent=2))
    print(f"  Saved → {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Step 4: save final edited w_r at α*
# --------------------------------------------------------------------------- #
def save_w_r_star(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                  *, output_dir: Path) -> Path:
    out_path = output_dir / f"{rm_key}_w_r_star.pt"
    dirs_bundle = torch.load(output_dir / f"{rm_key}_directions.pt",
                              map_location="cpu", weights_only=False)
    w_r = dirs_bundle["w_r"].float()
    target_cats = dirs_bundle["target_categories"]
    V = torch.stack([dirs_bundle["v_dict"][c] for c in target_cats
                     if c in dirs_bundle["v_dict"]]).float()
    alpha_star = rms_cfg["models"][rm_key]["alpha_star"]
    w_a = ablate_w_r(w_r, V, alpha_star,
                      svd_threshold=harve_cfg["ablation"]["svd_threshold"])
    torch.save({
        "rm_key": rm_key,
        "alpha_star": alpha_star,
        "w_r": w_r,
        "w_r_alpha": w_a,
    }, out_path)
    print(f"  α* = {alpha_star}  →  saved {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True, help="key from configs/rms.yaml")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "harve"))
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--skip_sweep", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    harve_cfg = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text())
    set_seed(harve_cfg.get("seed", 20260506))

    if not args.skip_extract:
        if not (output_dir / f"{args.rm}_directions.pt").exists():
            extract_all_directions(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)
        else:
            print(f"[1/4] extraction artifact exists — skip")
    if not args.skip_cache:
        cache_test_hiddens(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)
    if not args.skip_sweep:
        alpha_sweep(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)
    save_w_r_star(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)
    print("\nHARVE pipeline complete for", args.rm)


if __name__ == "__main__":
    main()
