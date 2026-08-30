#!/usr/bin/env bash
# Fine-tuning baselines for the comparison table.
#
# Each RM gets LoRA-trained at two general:legal ratios (3:1 and 5:1) on
# RewardHackBench train + general pairs from Skywork-Reward-Preference-80K-v0.2.
# Then evaluates the trained model on RewardHackBench test + RM-Bench Hard.
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
  for r in 3 5; do
    echo "==================== Finetune: $rm  ratio ${r}:1 ===================="
    python -m src.finetune.train_finetune \
      --rm "$rm" --ratio "$r" \
      --output "runs/finetune/${rm}_r${r}"
  done
done

echo "Finetuning complete. Trained adapters in runs/finetune/."
echo "Note: to evaluate the fine-tuned adapters, merge them into the base"
echo "model and re-run src.eval.run_rm_baseline / run_rmbench against the"
echo "merged checkpoint."
