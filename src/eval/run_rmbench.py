"""RM-Bench Hard evaluation with optional HARVE editing.
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
    load_rm_for_caching,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Stage A: GPU caching
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_rmbench_hiddens(rm_key: str, *, output_dir: Path,
                          max_length: int = 1024) -> Path:
    """Forward the model over RM-Bench and cache (h, kind, pi, idx) tensors.

    Returns the cache.pt path. Idempotent.
    """
    cache_path = output_dir / f"{rm_key}_rmbench_cache.pt"
    if cache_path.exists():
        print(f"[rmbench] cache exists at {cache_path}, skipping forward pass")
        return cache_path

    from datasets import load_dataset
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    bench_cfg = yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text())
    spec = rms_cfg["models"][rm_key]

    model, tok = load_rm_for_caching(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )

    ds = load_dataset(bench_cfg["rmbench"]["hf_id"],
                       split=bench_cfg["rmbench"]["split"])
    print(f"[rmbench] {rm_key}: {len(ds)} prompts × 6 completions = {len(ds)*6}")

    flat_h, flat_kind, flat_pi, flat_idx, ds_meta = [], [], [], [], []
    for pi, ex in enumerate(tqdm(ds, desc="  caching")):
        ds_meta.append({"prompt_idx": pi, "domain": ex["domain"]})
        for kind, completions in (("chosen", ex["chosen"]), ("rejected", ex["rejected"])):
            for j, resp in enumerate(completions):
                enc = apply_chat_template_pair(tok, ex["prompt"], resp, max_length=max_length)
                ids = enc["input_ids"].to(model.device)
                mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                pos = -1 if getattr(tok, "padding_side", "right") == "left" else mask[0].sum().item() - 1
                flat_h.append(out.hidden_states[-1][0, pos, :].float().cpu())
                flat_kind.append(kind)
                flat_pi.append(pi)
                flat_idx.append(j)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "h": torch.stack(flat_h),
        "kind": flat_kind, "pi": flat_pi, "idx": flat_idx, "meta": ds_meta,
    }, cache_path)
    print(f"[rmbench] saved cache → {cache_path}")
    del model
    torch.cuda.empty_cache()
    return cache_path


# --------------------------------------------------------------------------- #
# Stage B: CPU scoring at any alpha
# --------------------------------------------------------------------------- #
def _compute_per_prompt(scores_by_prompt: list[dict]) -> list[dict]:
    metrics = []
    for entry in scores_by_prompt:
        c, r = entry["chosen"], entry["rejected"]
        if len(c) != 3 or len(r) != 3 or any(v is None for v in c + r):
            metrics.append({"normal": None, "easy": None, "hard": None, "overall": None})
            continue
        M = np.array([[float(c[i] > r[j]) for j in range(3)] for i in range(3)])
        diag = float(np.mean([M[i, i] for i in range(3)]))
        upper = float(np.mean([M[i, j] for i in range(3) for j in range(3) if i < j]))
        lower = float(np.mean([M[i, j] for i in range(3) for j in range(3) if i > j]))
        metrics.append({"normal": diag, "easy": lower, "hard": upper, "overall": float(np.mean(M))})
    return metrics


def score_at_alphas(rm_key: str, alphas: list[float],
                     *, output_dir: Path) -> dict:
    """Compute RM-Bench hard/easy/normal at each alpha from the cached
    hiddens + the saved direction matrix V from runs/harve/."""
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    harve_cfg = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text())

    cache = torch.load(output_dir / f"{rm_key}_rmbench_cache.pt",
                       map_location="cpu", weights_only=False)
    h = cache["h"].float()

    # Load w_r + V from the HARVE artifact (or load_rm_for_scoring as a fallback).
    harve_path = REPO_ROOT / "runs" / "harve" / f"{rm_key}_directions.pt"
    if harve_path.exists():
        dirs = torch.load(harve_path, map_location="cpu", weights_only=False)
        w_r = dirs["w_r"].float()
        V = torch.stack([dirs["v_dict"][c]
                         for c in dirs["target_categories"]
                         if c in dirs["v_dict"]]).float()
    else:
        # Allow α=0 evaluation without HARVE directions.
        spec = rms_cfg["models"][rm_key]
        _, _, w_r = load_rm_for_scoring(spec["hf_id"],
                                          model_class=spec["model_class"],
                                          trust_remote_code=spec.get("trust_remote_code", False),
                                          device="cpu")
        V = torch.zeros(0, w_r.shape[0])

    sweep = []
    for a in alphas:
        w_a = ablate_w_r(w_r, V, a,
                         svd_threshold=harve_cfg["ablation"]["svd_threshold"])
        scores = (h @ w_a).detach().numpy()
        scores_by_prompt = [{"chosen": [None]*3, "rejected": [None]*3}
                            for _ in range(len(cache["meta"]))]
        for s, k, pi, j in zip(scores, cache["kind"], cache["pi"], cache["idx"]):
            scores_by_prompt[pi][k][j] = float(s)
        per_prompt = _compute_per_prompt(scores_by_prompt)

        dom_idx = defaultdict(list)
        for i, m in enumerate(cache["meta"]):
            dom_idx[m["domain"]].append(i)
        dom_acc = {d: {k: float(np.mean([per_prompt[i][k] for i in idxs]))
                       for k in ("normal", "easy", "hard", "overall")}
                       | {"n": len(idxs)}
                   for d, idxs in dom_idx.items()}
        overall = {k: float(np.mean([per_prompt[i][k] for i in range(len(per_prompt))]))
                   for k in ("normal", "easy", "hard", "overall")}
        sweep.append({"alpha": a, "overall": overall, "domain_acc": dom_acc})
        print(f"  α={a:>5}  hard={overall['hard']*100:.2f}  overall={overall['overall']*100:.2f}")

    out_path = output_dir / f"{rm_key}_rmbench_sweep.json"
    out_path.write_text(json.dumps({"rm_key": rm_key, "alphas": alphas, "sweep": sweep}, indent=2))
    print(f"[rmbench] saved → {out_path}")
    return {"alphas": alphas, "sweep": sweep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "rmbench"))
    parser.add_argument("--alphas", default=None,
                        help="comma-separated. Default: use configs/harve.yaml::ablation.alphas")
    parser.add_argument("--skip_cache", action="store_true")
    args = parser.parse_args()
    set_seed(20260506)

    output_dir = Path(args.output_dir)
    if not args.skip_cache:
        cache_rmbench_hiddens(args.rm, output_dir=output_dir)
    if args.alphas is not None:
        alphas = [float(a) for a in args.alphas.split(",")]
    else:
        alphas = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text()
                                )["ablation"]["alphas"]
    score_at_alphas(args.rm, alphas, output_dir=output_dir)


if __name__ == "__main__":
    main()
