"""Evaluate the HARVE-edited reward head on the held-out test split.

This is the ONLY place the test split meets an edited head. By the time this
runs, alpha* is already fixed by src.harve.run_harve (selected on data/dev.json
under the RewardBench guard); nothing here feeds back into selection.

Because scoring is linear in the final hidden state -- score = h . w -- one
forward pass over the test split supports every reward head. We cache the test
hidden states once, then read out the baseline head (w_r) and the edited head
(w_r_alpha) from the same cache, which is why the comparison is exact rather
than two separately-sampled runs.

Emits the main table's headline columns, Target / Non-target, plus the per
parent-category and per-subcategory breakdowns. RM-Bench is a separate script
(src.eval.run_rmbench).

Usage:
  python -m src.eval.run_harve_eval --rm Skywork-Reward-V2-Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.eval.compute_metrics import (
    parent_category_summary,
    per_subcategory_summary,
    target_nontarget_summary,
)
from src.utils import (
    apply_chat_template_pair,
    load_benchmark,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def cache_test_hiddens(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                       *, output_dir: Path) -> Path:
    """Cache final-layer last-token hidden states for the test split. Idempotent."""
    out_path = output_dir / f"{rm_key}_test_cache.pt"
    if out_path.exists():
        print(f"[eval] {out_path.name} exists — skipping forward pass")
        return out_path

    spec = rms_cfg["models"][rm_key]
    model, tok, _ = load_rm_for_scoring(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    pairs = load_benchmark("test")
    max_length = harve_cfg.get("caching", {}).get("max_length")   # None = no truncation
    print(f"[eval] caching test hiddens for {rm_key}: {len(pairs)} pairs")

    cache: dict[tuple[str, int, str], torch.Tensor] = {}
    for i, r in enumerate(tqdm(pairs, desc="  caching")):
        for role, resp in [("gold", r["gold_response"]),
                           ("hacked", r["hacked_response"])]:
            enc = apply_chat_template_pair(tok, r["question"], resp,
                                           max_length=max_length)
            ids = enc["input_ids"].to(model.device)
            mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            pos = -1 if getattr(tok, "padding_side", "right") == "left" else mask[0].sum().item() - 1
            cache[(r["category"], i, role)] = out.hidden_states[-1][0, pos, :].float().cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"rm_key": rm_key, "split": "test",
                "cache": cache, "n_pairs": len(pairs)}, out_path)
    print(f"[eval] saved → {out_path}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def _score_rows(pairs: list[dict], cache: dict, w: torch.Tensor) -> list[dict]:
    rows = []
    for i, r in enumerate(pairs):
        h_g = cache[(r["category"], i, "gold")]
        h_h = cache[(r["category"], i, "hacked")]
        rows.append({"category": r["category"],
                     "gold_score": float(h_g @ w),
                     "hacked_score": float(h_h @ w)})
    return rows


def _report(name: str, rows: list[dict], target_cats: list[str]) -> dict:
    tn = target_nontarget_summary(rows, target_cats)
    par = parent_category_summary(rows)
    sub = per_subcategory_summary(rows)
    print(f"\n  {name}")
    print(f"    Target      {tn['target']['accuracy']:6.2f}%  "
          f"({tn['target']['correct']}/{tn['target']['n']})")
    print(f"    Non-target  {tn['non_target']['accuracy']:6.2f}%  "
          f"({tn['non_target']['correct']}/{tn['non_target']['n']})")
    print(f"    Overall     {tn['overall']['accuracy']:6.2f}%  "
          f"({tn['overall']['correct']}/{tn['overall']['n']})")
    print("    Parent:  " + "  ".join(
        f"{k}={par[k]['accuracy']:.1f}" for k in sorted(par) if k != "_overall"))
    return {"target_non_target": tn, "per_parent": par, "per_subcategory": sub}


def evaluate(rm_key: str, *, harve_dir: Path, output_dir: Path) -> dict:
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    harve_cfg = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text())

    star_path = harve_dir / f"{rm_key}_w_r_star.pt"
    if not star_path.exists():
        raise SystemExit(f"Missing {star_path}. Run\n"
                         f"  python -m src.harve.run_harve --rm {rm_key}\n"
                         f"first.")
    star = torch.load(star_path, map_location="cpu", weights_only=False)
    dirs = torch.load(harve_dir / f"{rm_key}_directions.pt",
                      map_location="cpu", weights_only=False)
    # The edited subspace, not the declared target list: a subcategory with too
    # few fooled train pairs contributes no direction and is not edited.
    target_cats = [c for c in dirs["target_categories"] if c in dirs["v_dict"]]

    cache_path = cache_test_hiddens(rm_key, rms_cfg, harve_cfg, output_dir=output_dir)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)["cache"]
    pairs = load_benchmark("test")

    alpha_star = float(star["alpha_star"])
    print(f"\n[eval] {rm_key}   α* = {alpha_star} ({star.get('alpha_star_source', '?')})")
    print(f"[eval] edited subcategories: {target_cats}")

    base = _report("Baseline (α = 0)", _score_rows(pairs, cache, star["w_r"].float()),
                   target_cats)
    edit = _report(f"HARVE (α = {alpha_star})",
                   _score_rows(pairs, cache, star["w_r_alpha"].float()), target_cats)

    d_t = (edit["target_non_target"]["target"]["accuracy"]
           - base["target_non_target"]["target"]["accuracy"])
    d_n = (edit["target_non_target"]["non_target"]["accuracy"]
           - base["target_non_target"]["non_target"]["accuracy"])
    print(f"\n  Δ Target {d_t:+.2f}pp   Δ Non-target {d_n:+.2f}pp")

    result = {"rm_key": rm_key, "alpha_star": alpha_star,
              "target_categories": target_cats,
              "baseline": base, "harve": edit,
              "delta_target_pp": d_t, "delta_non_target_pp": d_n}
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rm_key}_harve_test_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[eval] saved → {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rm", required=True, help="key from configs/rms.yaml")
    parser.add_argument("--harve_dir", default=str(REPO_ROOT / "runs" / "harve"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "eval"))
    args = parser.parse_args()
    set_seed(20260506)
    evaluate(args.rm, harve_dir=Path(args.harve_dir), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
