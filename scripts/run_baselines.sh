#!/usr/bin/env bash
# Run baseline RM evaluation on RewardHackBench (test split) + RM-Bench Hard
# for all 8 RMs listed in configs/rms.yaml.
#
# Outputs:
#   runs/baseline/<rm>_scored_test.json    — per-pair scores
#   runs/baseline/<rm>_metrics_test.json   — per-subcategory + per-parent
#   runs/rmbench/<rm>_rmbench_cache.pt     — cached RM-Bench hiddens
#   runs/rmbench/<rm>_rmbench_sweep.json   — α=0 entry only is the baseline
#
# Set HF_HUB_ENABLE_HF_TRANSFER=1 in your shell for faster HF downloads.
set -euo pipefail

cd "$(dirname "$0")/.."

RMS=(
  Skywork-Reward-V2-Qwen3-0.6B
  internlm2-1_8b-reward
  GRM-Llama3.2-3B-rewardmodel-ft
  Skywork-Reward-V2-Llama-3.2-3B
  RM-Mistral-7B
  FsfairX-LLaMA3-RM-v0.1
  Skywork-Reward-Llama-3.1-8B-v0.2
  internlm2-20b-reward
)

for rm in "${RMS[@]}"; do
  echo "==================== Baseline: $rm ===================="
  python -m src.eval.run_rm_baseline --rm "$rm" --split test
  python -m src.eval.run_rmbench    --rm "$rm" --alphas "0.0"
done

echo "Done. See runs/baseline and runs/rmbench."
