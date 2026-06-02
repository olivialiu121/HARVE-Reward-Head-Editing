"""Load RewardHackBench JSON splits, the taxonomy, and the RM-Bench dataset."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_benchmark_config() -> dict:
    cfg_path = REPO_ROOT / "configs" / "benchmark.yaml"
    return yaml.safe_load(cfg_path.read_text())


def load_benchmark(split: str) -> list[dict]:
    """Load one split of RewardHackBench. `split` ∈ {'train', 'dev', 'test'}."""
    cfg = _load_benchmark_config()
    rel = cfg["rewardhackbench"][split]
    pairs = json.loads((REPO_ROOT / rel).read_text())
    expected = cfg["rewardhackbench"]["n_pairs"][split]
    assert len(pairs) == expected, (
        f"{split}.json has {len(pairs)} pairs, expected {expected}"
    )
    return pairs


def load_taxonomy() -> dict:
    """Load the category taxonomy (parent + subcategory definitions)."""
    cfg = _load_benchmark_config()
    return json.loads((REPO_ROOT / cfg["rewardhackbench"]["taxonomy"]).read_text())


def iter_pairs_by_subcategory(
    pairs: Iterable[dict],
) -> dict[str, list[dict]]:
    """Group pairs by their `category` key (e.g., A2_legalese_padding)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in pairs:
        out[r["category"]].append(r)
    return dict(out)


def load_rmbench(split: str = "train") -> list[dict]:
    """Load THU-KEG/RM-Bench via the `datasets` library. Each row has 3
    chosen + 3 rejected completions for the same prompt; downstream code
    forms the 3×3 matrix per prompt and reports the upper-triangle ("hard")
    mean as the headline metric."""
    from datasets import load_dataset
    cfg = _load_benchmark_config()
    ds = load_dataset(cfg["rmbench"]["hf_id"], split=split)
    return list(ds)
