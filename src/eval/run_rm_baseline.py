"""Baseline RM evaluation on RewardHackBench.

For one RM (specified by --rm) and one split (default: test), score every
gold/hacked pair, save the per-pair scores to ``runs/baseline/{rm}_scored.json``,
and print per-subcategory / parent-category micro accuracies.
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
)
from src.utils import (
    apply_chat_template_pair,
    load_benchmark,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _score_pair(model, tok, question: str, response: str,
                *, max_length: int = 1024) -> float:
    enc = apply_chat_template_pair(tok, question, response, max_length=max_length)
    ids = enc["input_ids"].to(model.device)
    mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
    out = model(input_ids=ids, attention_mask=mask)
    # Standard HF SeqCls returns out.logits; some custom RMs return the score
    # tensor directly.
    scores = out.logits if hasattr(out, "logits") else out
    return float(scores.squeeze().detach().cpu().item())


def evaluate(rm_key: str, *, split: str = "test",
             output_dir: Path | None = None, max_length: int = 1024) -> dict:
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    spec = rms_cfg["models"][rm_key]
    print(f"\n[baseline] {rm_key}  ({spec['hf_id']})  split={split}")

    model, tok, _ = load_rm_for_scoring(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    pairs = load_benchmark(split)

    scored = []
    for r in tqdm(pairs, desc="  scoring"):
        s_g = _score_pair(model, tok, r["question"], r["gold_response"], max_length=max_length)
        s_h = _score_pair(model, tok, r["question"], r["hacked_response"], max_length=max_length)
        scored.append({**r,
                       "gold_score": s_g, "hacked_score": s_h,
                       "score_diff": s_g - s_h})

    sub = per_subcategory_summary(scored)
    par = parent_category_summary(scored)
    print(f"\n  Overall: {sub['_overall']['accuracy']:.2f}% ({sub['_overall']['correct']}/{sub['_overall']['n']})")
    print(f"  Per parent category:")
    for k in sorted(par):
        if k == "_overall":
            continue
        v = par[k]
        print(f"    {k}: {v['accuracy']:.2f}%  ({v['correct']}/{v['n']})")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        scored_path = output_dir / f"{rm_key}_scored_{split}.json"
        scored_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
        metrics_path = output_dir / f"{rm_key}_metrics_{split}.json"
        metrics_path.write_text(json.dumps({"per_subcategory": sub,
                                             "per_parent": par}, indent=2))
        print(f"  Wrote {scored_path.name} + {metrics_path.name}")
    return {"per_subcategory": sub, "per_parent": par}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "baseline"))
    args = parser.parse_args()
    set_seed(20260506)
    evaluate(args.rm, split=args.split, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
