#!/usr/bin/env bash
# End-to-end HARVE pipeline for all 8 RMs.
#
# src.harve.run_harve performs, in order:
#   1. extract per-subcategory v_k from the train.json fooled subset
#   2. cache DEV-split hidden states
#   3. α-sweep on dev (Target resistance) — the selection objective
#   4. α-sweep on the held-out general RewardBench guard (RewardBench minus the
#      reported LLMBar subsets) — the selection constraint
#   5. select α* = argmax dev Target s.t. RewardBench drop ≤ 4pp (ties → smallest α)
#   6. save the edited head at α*
# It never reads data/test.json. The guard sweep runs BEFORE selection, because
# selection consumes it.
#
# Then, with α* already fixed:
#   - src.eval.run_harve_eval : held-out test split, Target / Non-target columns
#   - src.eval.run_rmbench    : RM-Bench Hard, reported held-out capability
#                               (swept across α for the tradeoff figure, but
#                               never used to select α*)
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
  # Steps 1-6, including the RewardBench guard sweep it needs for selection.
  python -m src.harve.run_harve --rm "$rm"

  # Held-out evaluation, after α* is fixed.
  python -m src.eval.run_harve_eval --rm "$rm"

  # RM-Bench Hard: reported held-out capability (main table + tradeoff figure),
  # scored across the α grid but NOT used to select α*.
  python -m src.eval.run_rmbench --rm "$rm"
done

echo "All RMs processed."
echo "  runs/harve/{rm}_alpha_star.json          selected α* + the full feasible grid"
echo "  runs/harve/{rm}_w_r_star.pt              edited reward head"
echo "  runs/eval/{rm}_harve_test_metrics.json   Target / Non-target on test"
