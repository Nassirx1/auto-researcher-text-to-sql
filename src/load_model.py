from __future__ import annotations

from typing import Any


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(str(dtype_name).lower(), torch.float16)


def load_base_model(config: dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Model loading requires torch, transformers, accelerate, and optionally bitsandbytes. "
            "Install requirements.txt before baseline/training runs."
        ) from exc

    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id")
    if not model_id:
        raise ValueError("No model.model_id configured. Use a model config for baseline/training runs.")
    dtype = _torch_dtype(model_cfg.get("torch_dtype", "float16"))
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
    }
    if model_cfg.get("load_in_4bit", False):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_model_with_adapter(config: dict[str, Any], adapter_path: str):
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Adapter loading requires peft. Install requirements.txt first.") from exc
    model, tokenizer = load_base_model(config)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer
