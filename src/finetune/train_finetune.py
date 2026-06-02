"""LoRA-based fine-tuning baseline.

LoRA fine-tuning of a reward model on RewardHackBench train 
pairs (legal-domain only) mixed with general preference pairs from
Skywork-Reward-Preference-80K-v0.2. Configurable general:legal ratio (default
3:1 and 5:1, as reported in the paper).

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from src.utils.seed import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_messages(prompt: str, response: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _load_legal_pairs(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    return [{
        "chosen": _to_messages(r["question"], r["gold_response"]),
        "rejected": _to_messages(r["question"], r["hacked_response"]),
    } for r in rows]


def _load_general_pairs(dataset_name: str, n: int, seed: int) -> list[dict]:
    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"chosen": r["chosen"], "rejected": r["rejected"]} for r in ds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm", required=True, help="key from configs/rms.yaml")
    parser.add_argument("--ratio", required=True, type=int,
                        help="general:legal ratio (e.g., 3 or 5)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rms_cfg = yaml.safe_load((REPO_ROOT / "configs" / "rms.yaml").read_text())
    ft_cfg = yaml.safe_load((REPO_ROOT / "configs" / "finetune.yaml").read_text())

    spec = rms_cfg["models"][args.rm]
    print(f"[finetune] {args.rm} ratio={args.ratio}:1  base={spec['hf_id']}")
    set_seed(ft_cfg["seed"])

    legal_train = _load_legal_pairs(REPO_ROOT / "data" / "train.json")
    legal_dev = _load_legal_pairs(REPO_ROOT / "data" / "dev.json")
    n_general = len(legal_train) * args.ratio
    print(f"  legal train: {len(legal_train)}  general: {n_general}")
    general = _load_general_pairs(ft_cfg["general_dataset"], n_general, ft_cfg["seed"])

    train_data = legal_train + general
    print(f"  combined train: {len(train_data)} pairs")

    tok = AutoTokenizer.from_pretrained(spec["hf_id"],
                                         trust_remote_code=spec.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(legal_dev)

    lora_kwargs = dict(ft_cfg["lora"])
    peft_cfg = LoraConfig(**lora_kwargs)

    model = AutoModelForSequenceClassification.from_pretrained(
        spec["hf_id"], num_labels=1, torch_dtype=torch.bfloat16,
        trust_remote_code=spec.get("trust_remote_code", False),
    )
    model.config.pad_token_id = tok.pad_token_id

    rc_kwargs = dict(ft_cfg["training"])
    train_args = RewardConfig(output_dir=args.output, **rc_kwargs)
    trainer = RewardTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tok,
        peft_config=peft_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    print(f"[finetune] saved adapter to {args.output}")


if __name__ == "__main__":
    main()
