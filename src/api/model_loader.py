import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from src.utils.config import LORA_CONFIGS, MODEL_PATHS

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INT = 0 if DEVICE == "cuda" else -1   # pipeline() usa -1 para CPU


def _load_causal(model_id: str, lora_path: str):
    if not Path(lora_path).exists():
        logger.warning("[LOADER] Caminho não encontrado: %s — pulando.", lora_path)
        return None

    logger.info("[LOADER] Carregando causal: %s + LoRA em %s", model_id, lora_path)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.to(DEVICE)
    model.eval()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE_INT,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )


def _load_seq2seq(model_id: str, lora_path: str):
    if not Path(lora_path).exists():
        logger.warning("[LOADER] Caminho não encontrado: %s — pulando.", lora_path)
        return None

    logger.info("[LOADER] Carregando seq2seq: %s + LoRA em %s", model_id, lora_path)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    base = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.to(DEVICE)
    model.eval()

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE_INT,
    )


def load_all_models() -> dict:
    models = {}

    models["causal-gpt2-medium"] = _load_causal(
        model_id  = LORA_CONFIGS["causal_1"]["model_id"],
        lora_path = LORA_CONFIGS["causal_1"]["save_dir"],
    )

    models["causal-opt-1.3b"] = _load_causal(
        model_id  = LORA_CONFIGS["causal_2"]["model_id"],
        lora_path = LORA_CONFIGS["causal_2"]["save_dir"],
    )

    models["seq2seq-flan-t5"] = _load_seq2seq(
        model_id  = LORA_CONFIGS["seq2seq_1"]["model_id"],
        lora_path = LORA_CONFIGS["seq2seq_1"]["save_dir"],
    )

    models["seq2seq-bart-large"] = _load_seq2seq(
        model_id  = LORA_CONFIGS["seq2seq_2"]["model_id"],
        lora_path = LORA_CONFIGS["seq2seq_2"]["save_dir"],
    )

    loaded = [k for k, v in models.items() if v is not None]
    logger.info("[LOADER] Modelos carregados: %s", loaded)
    return models
