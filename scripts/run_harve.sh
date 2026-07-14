#!/usr/bin/env bash
# End-to-end HARVE pipeline for all 8 RMs:
#   1. Extract per-subcategory v_k from train.json fooled subset
#   2. Cache test-set hidden states
#   3. α-sweep over RewardHackBench and general RewardBench — the α-selection capability
#      guard (RewardBench minus the reported LLMBar subsets); used to select α*
#   4. α-sweep over RM-Bench Hard — reported held-out capability (main table +
#      tradeoff figure)
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

  # α-selection capability guard: held-out general RewardBench subsets
  # (RewardBench minus the reported LLMBar subsets). α* is kept within
  # configs/harve.yaml::alpha_star_selection.constraint_delta of the unedited
  # model on this set; RM-Bench is never used for selection.
  python -m src.eval.run_rewardbench --rm "$rm"

  # RM-Bench Hard: reported held-out capability (main table + tradeoff figure),
  # scored across the α grid but NOT used to select α*.
  python -m src.eval.run_rmbench --rm "$rm"
done

echo "All RMs processed. See runs/harve/{rm}_w_r_star.pt for edited heads."
