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
│   ├── eval/                Baseline evaluation, RM-Bench evaluation, and metric computation
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

1. extracts hacking directions from fooled examples;
2. constructs the hacking subspace;
3. edits the reward-head vector;
4. evaluates the edited reward model on the held-out test split.

Equivalent direct command for one reward model:

```bash
python -m src.harve.run_harve --rm Skywork-Reward-V2-Qwen3-0.6B
```

This runs direction extraction, test-cache construction, alpha sweep, and reward-head editing for one model. The edited reward-head vector selected at `alpha_star` is saved in `runs/harve/`.

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
| `configs/harve.yaml` | HARVE extraction, SVD threshold, and alpha settings |
| `configs/finetune.yaml` | Fine-tuning baseline hyperparameters |