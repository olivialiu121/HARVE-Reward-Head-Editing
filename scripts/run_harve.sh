#!/usr/bin/env bash
# End-to-end HARVE pipeline for all 8 RMs:
#   1. Extract per-subcategory v_k from train.json fooled subset
#   2. Cache test-set hidden states
#   3. α-sweep over RewardHackBench
#   4. α-sweep over RM-Bench Hard (re-uses GPU cache built by run_baselines.sh)
#   5. Save final edited w_r at α*
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
  echo "==================== HARVE: $rm ===================="
  python -m src.harve.run_harve --rm "$rm"

  # Score RM-Bench Hard across the same alpha grid using the same cache.
  python -m src.eval.run_rmbench --rm "$rm"
done

echo "All RMs processed. See runs/harve/{rm}_w_r_star.pt for edited heads."
