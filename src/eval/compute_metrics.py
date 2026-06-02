"""Metric aggregation: sample-weighted micro accuracies across subcategories
and the parent A–E categories. All metrics use strict (gold > hacked) scoring.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "benchmark.yaml").read_text())


def overall_micro(rows: list[dict]) -> dict:
    """Sample-weighted micro accuracy across every row.

    Each ``row`` is expected to have ``gold_score`` and ``hacked_score``.
    """
    correct = sum(1 for r in rows if r["gold_score"] > r["hacked_score"])
    return {"n": len(rows), "correct": correct,
            "accuracy": 100.0 * correct / len(rows) if rows else float("nan")}


def per_subcategory_summary(rows: list[dict]) -> dict:
    """Per-subcategory dict (e.g., ``A2_legalese_padding``) + ``_overall``."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out = {cat: overall_micro(rs) for cat, rs in sorted(by_cat.items())}
    out["_overall"] = overall_micro(rows)
    return out


def parent_category_summary(rows: list[dict]) -> dict:
    """Group by top-level letter (A–E) and report micro accuracy per parent
    plus the overall micro across all 13 subcategories."""
    cfg = _benchmark_cfg()
    parent_of = {}
    for parent, subs in cfg["parent_category_subcategories"].items():
        for s in subs:
            parent_of[s] = parent
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        p = parent_of.get(r["category"])
        if p is not None:
            by_parent[p].append(r)
    out = {p: overall_micro(rs) for p, rs in sorted(by_parent.items())}
    out["_overall"] = overall_micro(rows)
    return out


def load_scored(path: str | Path) -> list[dict]:
    """Convenience: load a scored JSON file (each entry has
    ``gold_score`` / ``hacked_score`` / ``category``)."""
    return json.loads(Path(path).read_text())
