"""Direction extraction (HARVE step 1).

For each target subcategory of an RM we cache last-token hidden states for the
gold and hacked responses of every pair in the extraction split (train.json),
then compute v_k = mean(h_hacked - h_gold) over the *fooled* subset of those
pairs (pairs where the base RM already preferred hacked over gold).

We pool the per-subcategory v_k's into a per-RM matrix V ∈ R^{k×d}, optionally
unit-normalize, and pass V to ``edit_reward_head.build_ablation_basis``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.utils import (
    apply_chat_template_pair,
    iter_pairs_by_subcategory,
    load_benchmark,
    load_rm_for_caching,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _hidden_state(model, tok, question: str, response: str,
                  *, max_length: int = 1024) -> torch.Tensor:
    """Forward one (question, response) and return the final-layer last-token
    hidden state as float32 on CPU."""
    enc = apply_chat_template_pair(tok, question, response, max_length=max_length)
    ids = enc["input_ids"].to(model.device)
    mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    # Last-token pooling (or right-most non-pad if padding is on the left).
    pos = -1 if getattr(tok, "padding_side", "right") == "left" else mask[0].sum().item() - 1
    return out.hidden_states[-1][0, pos, :].float().cpu()


@torch.no_grad()
def _score(h: torch.Tensor, w_r: torch.Tensor) -> float:
    return float(torch.dot(h.float(), w_r.float()).item())


def extract_direction(model, tok, w_r: torch.Tensor, pairs: list[dict],
                      *, restrict_to_fooled: bool = True,
                      min_fooled: int = 3,
                      unit_norm: bool = True,
                      max_length: int = 1024) -> tuple[torch.Tensor | None, dict]:
    """Extract v_k for a single subcategory.

    Returns:
        v_k:  (d,) tensor or None if too few fooled pairs.
        info: dict with n_total / n_fooled / cos_with_w_r etc.
    """
    deltas, fooled_flags, n_total = [], [], 0
    for r in tqdm(pairs, desc="    extracting", leave=False):
        h_g = _hidden_state(model, tok, r["question"], r["gold_response"], max_length=max_length)
        h_h = _hidden_state(model, tok, r["question"], r["hacked_response"], max_length=max_length)
        delta = h_h - h_g
        s_g = _score(h_g, w_r)
        s_h = _score(h_h, w_r)
        deltas.append(delta)
        fooled_flags.append(s_h > s_g)
        n_total += 1

    fooled_indices = [i for i, f in enumerate(fooled_flags) if f]
    n_fooled = len(fooled_indices)
    info = {"n_total": n_total, "n_fooled": n_fooled}

    selected = fooled_indices if restrict_to_fooled else list(range(n_total))
    if restrict_to_fooled and n_fooled < min_fooled:
        info["skipped"] = f"only {n_fooled} fooled pairs (< {min_fooled})"
        return None, info

    v_k = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    if unit_norm:
        v_k = v_k / (v_k.norm() + 1e-8)

    info["cos_with_w_r"] = float(torch.nn.functional.cosine_similarity(
        v_k.unsqueeze(0), w_r.unsqueeze(0)
    ).item())
    return v_k, info


def extract_all_directions(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                            *, output_dir: Path) -> dict:
    """Run extraction for every target subcategory of one RM.

    Writes ``{output_dir}/{rm_key}_directions.pt`` with keys::
        v_dict     : {subcategory_key: v_k (d,)}
        w_r        : (d,) the base RM's score-head weight
        info       : per-subcategory n_total / n_fooled / cos
        target_categories : list of subcategory keys for this RM
    """
    set_seed(harve_cfg.get("seed", 20260506))
    spec = rms_cfg["models"][rm_key]
    target_cats = spec["target_categories"]
    print(f"\n[HARVE/extract] {rm_key}  →  targets = {target_cats}")

    print(f"  Loading model {spec['hf_id']} for caching + score head…")
    model, tok = load_rm_for_caching(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    _, _, w_r = load_rm_for_scoring(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
        device="cpu",   # we only need w_r; saves GPU memory
    )
    assert w_r is not None, f"Could not locate score-head weight for {rm_key}"
    print(f"  w_r shape={tuple(w_r.shape)}  hidden_dim={spec['hidden_dim']}")

    train_pairs = load_benchmark("train")
    by_cat = iter_pairs_by_subcategory(train_pairs)

    v_dict: dict[str, torch.Tensor] = {}
    info_per_cat: dict[str, dict] = {}
    for cat in target_cats:
        pairs = by_cat.get(cat, [])
        if not pairs:
            print(f"  [warn] no train pairs for {cat}, skipping")
            continue
        print(f"  Extracting {cat}  (n={len(pairs)})")
        v_k, info = extract_direction(
            model, tok, w_r, pairs,
            restrict_to_fooled=harve_cfg["extraction"]["restrict_to_fooled"],
            min_fooled=harve_cfg["extraction"]["min_fooled_per_subcat"],
            unit_norm=(harve_cfg["extraction"]["pooled_axis_norm"] == "unit"),
        )
        if v_k is not None:
            v_dict[cat] = v_k
        info_per_cat[cat] = info
        print(f"    n_total={info['n_total']}, n_fooled={info['n_fooled']}"
              + (f", cos(v_k,w_r)={info.get('cos_with_w_r', float('nan')):+.3f}"
                 if v_k is not None else f", SKIPPED: {info.get('skipped','')}"))

    out_path = output_dir / f"{rm_key}_directions.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "rm_key": rm_key,
        "hf_id": spec["hf_id"],
        "v_dict": v_dict,
        "w_r": w_r,
        "info": info_per_cat,
        "target_categories": target_cats,
    }, out_path)
    print(f"  Saved → {out_path}")
    return {"out_path": str(out_path), "v_dict": v_dict, "info": info_per_cat}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True, help="key from configs/rms.yaml")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "harve"))
    args = parser.parse_args()

    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    harve_cfg = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text())
    extract_all_directions(args.rm, rms_cfg, harve_cfg,
                            output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
