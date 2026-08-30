# HARVE: Hacking-Aware Reward-Head Vector Editing

This repository contains the code for the paper:

> HARVE: Hacking-Aware Reward-Head Vector Editing for Robust Reward Models

HARVE is a training-free method for mitigating reward hacking in reward models. It estimates hacking-related directions from contrastive gold--hacked examples, constructs a multi-directional hacking subspace, and edits only the final reward-head vector while leaving the transformer backbone unchanged.

## Data Release Status

The benchmark data will be released in this repository after minor formatting cleanup.

## Repository Layout

```text
HARVE-Reward-Head-Editing/
├── data/                    Benchmark data
├── configs/                 RM list, benchmark paths, HARVE and fine-tuning settings
├── src/
│   ├── eval/                Baseline / edited-head / RewardBench / RM-Bench evaluation and metrics
│   ├── finetune/            Fine-tuning baselines
│   ├── harve/               Direction extraction, reward-head editing, and HARVE pipeline
│   └── utils/               Data loaders, RM loaders, scoring utilities, and seeding
└── scripts/                 One-command scripts for each pipeline stage
```

## Environment Setup

```bash
conda create -n harve python=3.10
conda activate harve
pip install -r requirements.txt
```

## Data

The benchmark contains 1,203 matched gold--hacked response pairs across 13 subcategories.

| Split | Description | Size |
|---|---|---:|
| `train.json` | 20% professional-domain extraction split | 157 |
| `dev.json` | 10% professional-domain calibration split | 79 |
| `test.json` | 70% professional-domain pairs + all general-purpose pairs | 967 |

The held-out test split contains 548 professional-domain pairs and 419 general-purpose pairs.

See `data/README.md` for detailed information.

## Reward Models

The paper evaluates eight reward models spanning 0.6B--20B parameters. Their Hugging Face IDs and model-specific settings are listed in `configs/rms.yaml`.

All eight reward models are publicly available.

## Running Baseline Reward-Model Evaluation

To evaluate the original reward models on the benchmark:

```bash
bash scripts/run_baselines.sh
```

Equivalent direct command for one reward model:

```bash
python -m src.eval.run_rm_baseline --rm Skywork-Reward-V2-Qwen3-0.6B --split test
```

The `--rm` flag accepts any key from `configs/rms.yaml`. Outputs are saved in `runs/baseline/`.

## Running HARVE

To run the full HARVE pipeline:

```bash
bash scripts/run_harve.sh
```

This pipeline:

1. extracts hacking directions from fooled examples in `train.json`;
2. constructs the hacking subspace;
3. selects the editing strength `alpha*` on `dev.json`;
4. edits the reward-head vector at `alpha*`;
5. evaluates the edited reward model on the held-out test split.

Equivalent direct commands for one reward model:

```bash
python -m src.harve.run_harve     --rm Skywork-Reward-V2-Qwen3-0.6B   # steps 1-4
python -m src.eval.run_harve_eval --rm Skywork-Reward-V2-Qwen3-0.6B   # step 5
```

`run_harve` writes the selected operating point to `runs/harve/{rm}_alpha_star.json`
(including the full grid, so the constraint is auditable) and the edited head to
`runs/harve/{rm}_w_r_star.pt`. `run_harve_eval` writes the held-out Target /
Non-target numbers to `runs/eval/`.

### How `alpha*` is selected

```
alpha* = argmax  DevTarget(alpha)
         s.t.    RBGeneral(alpha) >= RBGeneral(0) - 4pp
         ties broken toward the smallest alpha
```

- **`DevTarget`** — micro accuracy (gold scored above hacked) on `dev.json`,
  restricted to the reward model's own target subcategories, i.e. the ones whose
  directions are actually edited. Maximizing dev *overall* instead selects a
  different `alpha*` for some models, so the restriction matters.
- **`RBGeneral`** — a general-capability guard: the RewardBench (filtered)
  subsets *excluding* the five LLMBar subsets. LLMBar is folded into `test.json`
  as the off-topic / style (D/E) categories, so excluding it keeps the guard
  disjoint from everything the paper reports.

`src/harve/run_harve.py` never reads `data/test.json`. Neither the test split nor
RM-Bench is consulted during selection; both are reported fully held out. The
`alpha_star` values recorded in `configs/rms.yaml` are the paper's operating
points — the pipeline re-derives them and prints a loud mismatch warning if what
it selects disagrees.

## Running Fine-Tuning Baselines

To run the data-augmented fine-tuning baselines:

```bash
bash scripts/run_finetune.sh
```

The fine-tuning baselines use mixtures of general preference data and benchmark training pairs with 3:1 and 5:1 general-to-benchmark ratios. Hyperparameters are specified in `configs/finetune.yaml`.

## Configuration Files

| File | Description |
|---|---|
| `configs/rms.yaml` | Reward model names, Hugging Face IDs, and model-specific settings |
| `configs/benchmark.yaml` | Benchmark paths, split metadata, and category information |
| `configs/harve.yaml` | HARVE extraction, SVD threshold, alpha grid, and the `alpha*` selection rule |
| `configs/finetune.yaml` | Fine-tuning baseline hyperparameters |