"""Held-out general-RewardBench evaluation across the α grid.

This is the capability guard used for HARVE α* selection. It scores the
RewardBench (allenai/reward-bench, filtered) subsets that are NOT reported —
i.e. every subset except the five LLMBar ones that are folded into
RewardHackBench as the off_topic/style (D/E) columns. The remaining
Chat / Chat-Hard / Safety / Reasoning subsets are disjoint from RM-Bench and
from every reported metric, so RM-Bench and the legal test split remain fully
held-out (never consulted during α selection).

After the sweep it reads the per-RM alpha_star from configs/rms.yaml and the
constraint_delta from configs/harve.yaml, and reports whether the operating
point satisfies the general-capability guard (drop ≤ |constraint_delta| pp).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.harve.edit_reward_head import ablate_w_r
from src.utils import (
    apply_chat_template_pair,
    load_rewardbench_general,
    load_rm_for_caching,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

# Standard RewardBench subset → top-level category (filtered split).
SUBSET_TO_CATEGORY = {
    "alpacaeval-easy": "Chat", "alpacaeval-length": "Chat", "alpacaeval-hard": "Chat",
    "mt-bench-easy": "Chat", "mt-bench-med": "Chat",
    "mt-bench-hard": "Chat Hard",
    "refusals-dangerous": "Safety", "refusals-offensive": "Safety",
    "xstest-should-refuse": "Safety", "xstest-should-respond": "Safety",
    "donotanswer": "Safety",
    "math-prm": "Reasoning", "hep-cpp": "Reasoning", "hep-go": "Reasoning",
    "hep-java": "Reasoning", "hep-js": "Reasoning", "hep-python": "Reasoning",
    "hep-rust": "Reasoning",
}


def _accuracy(w, h_c, h_r, subsets) -> dict:
    correct = (h_c @ w > h_r @ w).numpy()
    by_cat = defaultdict(list)
    for i, s in enumerate(subsets):
        cat = SUBSET_TO_CATEGORY.get(s)
        if cat is not None:
            by_cat[cat].append(correct[i])
    cat_acc = {c: float(np.mean(v)) * 100 for c, v in by_cat.items()}
    return {
        "micro": float(correct.mean()) * 100,          # over all general pairs (guard metric)
        "category_macro": float(np.mean(list(cat_acc.values()))),
        "category_acc": cat_acc,
    }


@torch.no_grad()
def run_rewardbench(rm_key: str, alphas: list[float], *, output_dir: Path,
                    max_length: int = 1024) -> dict:
    """Forward once, then score every α. Writes {rm}_rewardbench_sweep.json and
    prints whether the operating point α* satisfies the general-capability guard."""
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    harve_cfg = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text())
    spec = rms_cfg["models"][rm_key]

    # --- forward pass (GPU): collect chosen/rejected hiddens in memory ---
    model, tok = load_rm_for_caching(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    pairs = load_rewardbench_general()
    n_subs = len({p["subset"] for p in pairs})
    print(f"[rewardbench] {rm_key}: {len(pairs)} held-out general pairs across {n_subs} subsets")

    def hidden(prompt: str, response: str) -> torch.Tensor:
        enc = apply_chat_template_pair(tok, prompt, response, max_length=max_length)
        ids = enc["input_ids"].to(model.device)
        mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        pos = -1 if getattr(tok, "padding_side", "right") == "left" else mask[0].sum().item() - 1
        return out.hidden_states[-1][0, pos, :].float().cpu()

    h_chosen, h_rejected, subsets = [], [], []
    for r in tqdm(pairs, desc="  forward"):
        h_chosen.append(hidden(r["prompt"], r["chosen"]))
        h_rejected.append(hidden(r["prompt"], r["rejected"]))
        subsets.append(r["subset"])
    h_c = torch.stack(h_chosen)
    h_r = torch.stack(h_rejected)
    del model
    torch.cuda.empty_cache()

    # --- reward head + direction basis (from the HARVE artifact) ---
    harve_path = REPO_ROOT / "runs" / "harve" / f"{rm_key}_directions.pt"
    if harve_path.exists():
        dirs = torch.load(harve_path, map_location="cpu", weights_only=False)
        w_r = dirs["w_r"].float()
        V = torch.stack([dirs["v_dict"][c] for c in dirs["target_categories"]
                         if c in dirs["v_dict"]]).float()
    else:
        _, _, w_r = load_rm_for_scoring(spec["hf_id"], model_class=spec["model_class"],
                                        trust_remote_code=spec.get("trust_remote_code", False),
                                        device="cpu")
        V = torch.zeros(0, w_r.shape[0])   # α=0 only (no editing)

    # --- score every α (CPU dot products) ---
    sweep = []
    for a in alphas:
        w_a = ablate_w_r(w_r, V, a, svd_threshold=harve_cfg["ablation"]["svd_threshold"])
        acc = _accuracy(w_a, h_c, h_r, subsets)
        sweep.append({"alpha": a, **acc})
        print(f"  α={a:>5}  general(micro)={acc['micro']:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rm_key}_rewardbench_sweep.json"
    out_path.write_text(json.dumps({"rm_key": rm_key, "alphas": alphas, "sweep": sweep}, indent=2))
    print(f"[rewardbench] saved → {out_path}")

    # --- guard check: does α* keep the general drop within budget? ---
    base = next(s["micro"] for s in sweep if s["alpha"] == 0.0)
    alpha_star = spec.get("alpha_star")
    delta = harve_cfg["alpha_star_selection"]["constraint_delta"]  # e.g. -4.0
    at_star = next((s["micro"] for s in sweep if abs(s["alpha"] - alpha_star) < 1e-9), None)
    if at_star is not None:
        drop = base - at_star
        ok = drop <= -delta + 1e-9
        print(f"[rewardbench] α*={alpha_star}: general drop = {drop:+.2f}pp "
              f"(guard ≤ {-delta:.1f}pp) → {'OK' if ok else 'VIOLATED'}")
    return {"alphas": alphas, "sweep": sweep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "rewardbench"))
    parser.add_argument("--alphas", default=None,
                        help="comma-separated. Default: configs/harve.yaml::ablation.alphas")
    args = parser.parse_args()
    set_seed(20260506)

    if args.alphas is not None:
        alphas = [float(a) for a in args.alphas.split(",")]
    else:
        alphas = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text()
                                )["ablation"]["alphas"]
    run_rewardbench(args.rm, alphas, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
