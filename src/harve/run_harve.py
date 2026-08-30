"""End-to-end HARVE pipeline for one RM.

This module never reads ``data/test.json``. The held-out split is touched only
by ``src.eval.run_harve_eval``, after alpha* has already been fixed.

    1. Extract per-subcategory directions from train.json  -> {rm}_directions.pt
    2. Cache dev-split hidden states                       -> {rm}_dev_cache.pt
    3. Sweep alpha on the dev split (Target resistance)    -> {rm}_dev_sweep.json
    4. Sweep alpha on the general RewardBench guard        -> {rm}_rewardbench_sweep.json
    5. Select alpha*                                       -> {rm}_alpha_star.json
    6. Apply alpha* and save the edited head               -> {rm}_w_r_star.pt

Selection rule (configs/harve.yaml::alpha_star_selection)::

    alpha* = argmax_alpha  DevTarget(alpha)
             s.t.          RBGeneral(alpha) >= RBGeneral(0) - 4pp
             ties broken toward the smallest alpha

``DevTarget`` is the sample-weighted micro accuracy (gold scored above hacked)
over the RM's own target subcategories -- the edited subspace, not all 13 -- on
data/dev.json. ``RBGeneral`` is a held-out general-capability guard: the
RewardBench subsets that are not reported anywhere (RewardBench-filtered minus
the five LLMBar subsets, which are folded into test.json as the D/E columns).
RM-Bench and the legal test split are never consulted.

The alpha* this script selects is compared against the paper value recorded in
configs/rms.yaml and any mismatch is reported loudly.

The script is idempotent: each step is skipped if its output already exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.harve.edit_reward_head import ablate_w_r
from src.harve.extract_directions import extract_all_directions
from src.utils import (
    apply_chat_template_pair,
    load_benchmark,
    load_rm_for_scoring,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

SELECTION_SPLIT = "dev"   # hard-coded: alpha* is never selected on test


def _edit_basis(dirs_bundle: dict) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(w_r, V, edited_categories) from a saved extraction bundle.

    A target subcategory with too few fooled train pairs produces no direction
    and is therefore absent from ``v_dict``; it is not part of the subspace.
    """
    w_r = dirs_bundle["w_r"].float()
    v_dict = dirs_bundle["v_dict"]
    cats = [c for c in dirs_bundle["target_categories"] if c in v_dict]
    V = (torch.stack([v_dict[c] for c in cats]).float() if cats
         else torch.zeros(0, w_r.shape[0]))
    return w_r, V, cats


# --------------------------------------------------------------------------- #
# Step 2: cache dev-split hidden states
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_dev_hiddens(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                      *, output_dir: Path) -> Path:
    out_path = output_dir / f"{rm_key}_{SELECTION_SPLIT}_cache.pt"
    if out_path.exists():
        print(f"  [cache] {out_path.name} exists — skipping")
        return out_path

    spec = rms_cfg["models"][rm_key]
    model, tok, _ = load_rm_for_scoring(
        spec["hf_id"],
        model_class=spec["model_class"],
        trust_remote_code=spec.get("trust_remote_code", False),
    )

    pairs = load_benchmark(SELECTION_SPLIT)
    max_length = harve_cfg.get("caching", {}).get("max_length")   # None = no truncation
    print(f"  Caching {SELECTION_SPLIT} hiddens for {rm_key}: {len(pairs)} pairs")
    cache: dict[tuple[str, int, str], torch.Tensor] = {}
    for i, r in enumerate(tqdm(pairs, desc="    caching")):
        for role, resp in [("gold", r["gold_response"]),
                           ("hacked", r["hacked_response"])]:
            enc = apply_chat_template_pair(tok, r["question"], resp,
                                           max_length=max_length)
            ids = enc["input_ids"].to(model.device)
            mask = enc.get("attention_mask", torch.ones_like(ids)).to(model.device)
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            pos = -1 if getattr(tok, "padding_side", "right") == "left" else mask[0].sum().item() - 1
            cache[(r["category"], i, role)] = out.hidden_states[-1][0, pos, :].float().cpu()

    torch.save({"rm_key": rm_key, "split": SELECTION_SPLIT,
                "cache": cache, "n_pairs": len(pairs)}, out_path)
    print(f"  Saved → {out_path}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


# --------------------------------------------------------------------------- #
# Step 3: dev alpha-sweep (the selection objective)
# --------------------------------------------------------------------------- #
def dev_sweep(rm_key: str, rms_cfg: dict, harve_cfg: dict,
              *, output_dir: Path) -> Path:
    """Score DevTarget at every alpha on the grid, from the cached dev hiddens."""
    out_path = output_dir / f"{rm_key}_{SELECTION_SPLIT}_sweep.json"
    if out_path.exists():
        print(f"  [sweep] {out_path.name} exists — skipping")
        return out_path

    dirs_bundle = torch.load(output_dir / f"{rm_key}_directions.pt",
                             map_location="cpu", weights_only=False)
    cache = torch.load(output_dir / f"{rm_key}_{SELECTION_SPLIT}_cache.pt",
                       map_location="cpu", weights_only=False)["cache"]
    w_r, V, edit_cats = _edit_basis(dirs_bundle)

    pairs = load_benchmark(SELECTION_SPLIT)
    # Rows of the objective: only the RM's own target subcategories.
    rows = [(i, r["category"]) for i, r in enumerate(pairs)
            if r["category"] in edit_cats]
    if not rows:
        raise SystemExit(f"{rm_key}: no dev pairs in the target subcategories "
                         f"{edit_cats} — cannot select alpha*.")
    print(f"  Dev objective: {len(rows)} pairs over {len(edit_cats)} target "
          f"subcategories {edit_cats}")

    svd_threshold = harve_cfg["ablation"]["svd_threshold"]
    sweep = []
    for a in harve_cfg["ablation"]["alphas"]:
        w_a = ablate_w_r(w_r, V, a, svd_threshold=svd_threshold)
        per_cat: dict[str, dict] = {}
        for i, cat in rows:
            e = per_cat.setdefault(cat, {"n": 0, "correct": 0})
            e["n"] += 1
            if float(cache[(cat, i, "gold")] @ w_a) > float(cache[(cat, i, "hacked")] @ w_a):
                e["correct"] += 1
        n = sum(e["n"] for e in per_cat.values())
        c = sum(e["correct"] for e in per_cat.values())
        sweep.append({"alpha": a, "dev_target": 100.0 * c / n,
                      "n": n, "correct": c, "per_cat": per_cat})
        print(f"    α={a:>5}  dev Target={sweep[-1]['dev_target']:.2f}%")

    out_path.write_text(json.dumps({
        "rm_key": rm_key, "split": SELECTION_SPLIT,
        "target_categories": edit_cats,
        "alphas": harve_cfg["ablation"]["alphas"], "sweep": sweep,
    }, indent=2))
    print(f"  Saved → {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Step 5: select alpha*
# --------------------------------------------------------------------------- #
def select_alpha_star(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                      *, output_dir: Path, rb_dir: Path) -> Path:
    """argmax DevTarget s.t. RBGeneral >= RBGeneral(0) - tol; ties -> smallest alpha."""
    out_path = output_dir / f"{rm_key}_alpha_star.json"

    dev = json.loads((output_dir / f"{rm_key}_{SELECTION_SPLIT}_sweep.json").read_text())
    rb_path = rb_dir / f"{rm_key}_rewardbench_sweep.json"
    if not rb_path.exists():
        raise SystemExit(
            f"Missing the capability guard sweep ({rb_path}). Run\n"
            f"  python -m src.eval.run_rewardbench --rm {rm_key}\n"
            f"before selecting alpha*.")
    rb = json.loads(rb_path.read_text())

    tol = abs(float(harve_cfg["alpha_star_selection"]["constraint_delta"]))
    dev_at = {round(float(s["alpha"]), 6): s["dev_target"] for s in dev["sweep"]}
    rb_at = {round(float(s["alpha"]), 6): s["micro"] for s in rb["sweep"]}

    grid = [a for a in harve_cfg["ablation"]["alphas"]
            if round(float(a), 6) in dev_at and round(float(a), 6) in rb_at]
    missing = [a for a in harve_cfg["ablation"]["alphas"]
               if round(float(a), 6) not in dev_at or round(float(a), 6) not in rb_at]
    if missing:
        print(f"  [warn] alphas missing from one of the sweeps, excluded: {missing}")
    if round(0.0, 6) not in [round(float(a), 6) for a in grid]:
        raise SystemExit(f"{rm_key}: alpha=0 must be on the grid (it defines the "
                         f"RewardBench reference point).")

    rb0 = rb_at[round(0.0, 6)]
    rows = []
    for a in grid:                       # grid stays in ascending config order
        k = round(float(a), 6)
        feasible = rb_at[k] >= rb0 - tol - 1e-9
        rows.append({"alpha": a, "dev_target": dev_at[k], "rb_general": rb_at[k],
                     "rb_drop": rb0 - rb_at[k], "feasible": feasible})

    feasible = [r for r in rows if r["feasible"]]
    # max() returns the FIRST maximizer, and the grid is ascending -> smallest alpha.
    best = max(feasible, key=lambda r: r["dev_target"])

    paper = rms_cfg["models"][rm_key].get("alpha_star")
    agrees = paper is not None and abs(float(paper) - float(best["alpha"])) < 1e-9
    result = {
        "rm_key": rm_key,
        "alpha_star": best["alpha"],
        "rule": ("argmax dev Target over the RM's target subcategories, s.t. "
                 f"RewardBench(general) >= RewardBench(0) - {tol:.1f}pp, "
                 "ties -> smallest alpha"),
        "selection_split": SELECTION_SPLIT,
        "constraint_tolerance_pp": tol,
        "rb_general_at_zero": rb0,
        "dev_target_at_star": best["dev_target"],
        "rb_general_at_star": best["rb_general"],
        "rb_drop_at_star": best["rb_drop"],
        "n_feasible": len(feasible),
        "paper_alpha_star": paper,
        "matches_paper": agrees,
        "grid": rows,
    }
    out_path.write_text(json.dumps(result, indent=2))

    print(f"  RewardBench(general) at α=0: {rb0:.2f}%  (budget {tol:.1f}pp → "
          f"floor {rb0 - tol:.2f}%)")
    print(f"  Feasible α: {[r['alpha'] for r in feasible]}")
    print(f"  α* = {best['alpha']}  dev Target={best['dev_target']:.2f}%  "
          f"RB drop={best['rb_drop']:+.2f}pp")
    if paper is None:
        print(f"  [note] configs/rms.yaml records no alpha_star for {rm_key}.")
    elif not agrees:
        print(f"  [MISMATCH] configs/rms.yaml records alpha_star={paper} but the "
              f"selection rule picks {best['alpha']}. Investigate before "
              f"reporting — do not silently overwrite either value.")
    print(f"  Saved → {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Step 6: save the edited head at alpha*
# --------------------------------------------------------------------------- #
def save_w_r_star(rm_key: str, rms_cfg: dict, harve_cfg: dict,
                  *, output_dir: Path) -> Path:
    out_path = output_dir / f"{rm_key}_w_r_star.pt"
    dirs_bundle = torch.load(output_dir / f"{rm_key}_directions.pt",
                             map_location="cpu", weights_only=False)
    w_r, V, edit_cats = _edit_basis(dirs_bundle)

    sel_path = output_dir / f"{rm_key}_alpha_star.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        alpha_star, source = float(sel["alpha_star"]), "selected on dev"
    else:
        # Fall back to the recorded paper value (e.g. --skip_select reruns).
        alpha_star = float(rms_cfg["models"][rm_key]["alpha_star"])
        source = "configs/rms.yaml (selection artifact absent)"

    w_a = ablate_w_r(w_r, V, alpha_star,
                     svd_threshold=harve_cfg["ablation"]["svd_threshold"])
    torch.save({
        "rm_key": rm_key,
        "alpha_star": alpha_star,
        "alpha_star_source": source,
        "edited_categories": edit_cats,
        "w_r": w_r,
        "w_r_alpha": w_a,
    }, out_path)
    print(f"  α* = {alpha_star} ({source})  →  saved {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rm", required=True, help="key from configs/rms.yaml")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "harve"))
    parser.add_argument("--rewardbench_dir",
                        default=str(REPO_ROOT / "runs" / "rewardbench"),
                        help="where the capability-guard sweep lives")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--skip_sweep", action="store_true")
    parser.add_argument("--skip_guard", action="store_true",
                        help="assume the RewardBench guard sweep already exists")
    parser.add_argument("--skip_select", action="store_true",
                        help="reuse the recorded alpha* instead of re-selecting")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rb_dir = Path(args.rewardbench_dir)
    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    harve_cfg = yaml.safe_load((REPO_ROOT / "configs" / "harve.yaml").read_text())
    set_seed(harve_cfg.get("seed", 20260506))

    print(f"[1/6] extract directions (train split)")
    if not args.skip_extract:
        if not (output_dir / f"{args.rm}_directions.pt").exists():
            extract_all_directions(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)
        else:
            print("  extraction artifact exists — skip")

    print(f"[2/6] cache {SELECTION_SPLIT}-split hidden states")
    if not args.skip_cache:
        cache_dev_hiddens(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)

    print(f"[3/6] α-sweep on the {SELECTION_SPLIT} split (selection objective)")
    if not args.skip_sweep:
        dev_sweep(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)

    print("[4/6] α-sweep on the held-out RewardBench capability guard")
    if not args.skip_guard:
        from src.eval.run_rewardbench import run_rewardbench
        run_rewardbench(args.rm, harve_cfg["ablation"]["alphas"], output_dir=rb_dir)

    print("[5/6] select α*")
    if not args.skip_select:
        select_alpha_star(args.rm, rms_cfg, harve_cfg,
                          output_dir=output_dir, rb_dir=rb_dir)

    print("[6/6] apply α* and save the edited head")
    save_w_r_star(args.rm, rms_cfg, harve_cfg, output_dir=output_dir)
    print(f"\nHARVE pipeline complete for {args.rm}. "
          f"Evaluate on the held-out split with:\n"
          f"  python -m src.eval.run_harve_eval --rm {args.rm}")


if __name__ == "__main__":
    main()
