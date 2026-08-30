"""Reward-model loading helpers.

Two main entry points:

* ``load_rm_for_caching``  — loads the base transformer body only (AutoModel)
  and returns ``(model, tokenizer)``. Used for the forward pass that caches
  last-token hidden states.

* ``load_rm_for_scoring`` — loads the full SequenceClassification model and
  returns ``(model, tokenizer, w_r)`` where ``w_r`` is the score-head weight
  ``(d,)`` extracted from the head. ``w_r`` is what HARVE edits.

"""
from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# Shim: transformers >=5.x removed DynamicCache.from_legacy_cache and
# .to_legacy_cache, but older custom modeling code (InternLM2) still calls
# both unconditionally. Restore them as no-ops if missing.
try:
    from transformers import DynamicCache as _DC
    if not hasattr(_DC, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            return cls()
        _DC.from_legacy_cache = _from_legacy_cache
    if not hasattr(_DC, "to_legacy_cache"):
        def _to_legacy_cache(self):
            return None
        _DC.to_legacy_cache = _to_legacy_cache
except Exception:
    pass


def load_patched_config(model_id: str, trust_remote_code: bool):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict):
        rope_type_val = rs.get("rope_type") or rs.get("type")
        if rope_type_val == "default":
            cfg.rope_scaling = None
        elif "type" not in rs and "rope_type" in rs:
            rs["type"] = rs["rope_type"]
    return cfg


def load_rm_for_caching(model_id: str,
                       *,
                       model_class: str = "auto",
                       trust_remote_code: bool = False,
                       attn_implementation: str | None = None,
                       device: str = "cuda"):
    """Load the transformer body for caching last-token hidden states.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(torch_dtype=torch.bfloat16)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if trust_remote_code:
        load_kwargs["config"] = load_patched_config(model_id, True)
        load_kwargs["trust_remote_code"] = True

    model = AutoModel.from_pretrained(model_id, **load_kwargs).to(device)
    model.eval()
    return model, tok


def load_rm_for_scoring(model_id: str,
                       *,
                       model_class: str = "auto_for_seq_cls",
                       trust_remote_code: bool = False,
                       attn_implementation: str | None = None,
                       device: str = "cuda"):
    """Load the full reward model (body + score head) and return its head
    weight ``w_r`` as a separate tensor. ``w_r`` is what HARVE edits."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(torch_dtype=torch.bfloat16)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    if model_class == "auto":
        if trust_remote_code:
            load_kwargs["config"] = load_patched_config(model_id, True)
            load_kwargs["trust_remote_code"] = True
        model = AutoModel.from_pretrained(model_id, **load_kwargs).to(device)
    else:
        load_kwargs["num_labels"] = 1
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **load_kwargs
        ).to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    # Best-effort: pull the score-head weight w_r from common attribute names.
    w_r = None
    for attr in ("score", "v_head", "value_head", "reward_head"):
        head = getattr(model, attr, None)
        if head is None:
            continue
        # head may be a Linear; try its weight
        weight = getattr(head, "weight", None)
        if weight is None and hasattr(head, "summary"):
            weight = getattr(head.summary, "weight", None)
        if weight is not None:
            w_r = weight.detach().float().squeeze(0).cpu()
            break
    if w_r is None and hasattr(model, "classifier"):
        weight = getattr(model.classifier, "weight", None)
        if weight is not None:
            w_r = weight.detach().float().squeeze(0).cpu()

    return model, tok, w_r
