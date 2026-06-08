import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from pathlib import Path
from peft import TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from src.training.lora_config import make_lora_config
from src.utils.helpers import build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _tokenize_causal(samples, tokenizer, max_len=512):
    texts = [build_prompt(s["Instruction"], s["Output"]) + tokenizer.eos_token
             for s in samples]
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return Dataset.from_dict(enc)

def _tokenize_seq2seq(samples, tokenizer, max_src=256, max_tgt=256):
    inputs  = [s["Instruction"] for s in samples]
    targets = [s["Output"]      for s in samples]
    enc = tokenizer(inputs, max_length=max_src, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        lbl = tokenizer(targets, max_length=max_tgt, truncation=True, padding="max_length")
    enc["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in row]
        for row in lbl["input_ids"]
    ]
    return Dataset.from_dict(enc)

def train_model(cfg: dict, samples: list[dict]) -> list[float]:
    """Treina um modelo (causal ou seq2seq) com LoRA e salva os adaptadores."""
    print(f"\n[TRAIN] {cfg['model_id']} → {cfg['save_dir']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    if cfg["model_type"] == "causal":
        base    = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=dtype)
        lora_cfg = make_lora_config(cfg, TaskType.CAUSAL_LM)
        model   = get_peft_model(base, lora_cfg)
        dataset = _tokenize_causal(samples, tokenizer)
        args    = TrainingArguments(
            output_dir=cfg["save_dir"], num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"], learning_rate=cfg["lr"],
            warmup_ratio=0.1, weight_decay=0.01, logging_steps=10,
            save_strategy="epoch", fp16=(DEVICE=="cuda"), report_to="none",
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
    else:
        base    = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_id"], torch_dtype=dtype)
        lora_cfg = make_lora_config(cfg, TaskType.SEQ_2_SEQ_LM)
        model   = get_peft_model(base, lora_cfg)
        dataset = _tokenize_seq2seq(samples, tokenizer)
        args    = Seq2SeqTrainingArguments(
            output_dir=cfg["save_dir"], num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"], learning_rate=cfg["lr"],
            warmup_ratio=0.1, weight_decay=0.01, logging_steps=10,
            save_strategy="epoch", predict_with_generate=True,
            fp16=(DEVICE=="cuda"), report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=model, args=args, train_dataset=dataset, tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model,
                                                  padding=True, label_pad_token_id=-100),
        )

    model.print_trainable_parameters()
    trainer.train()
    model.save_pretrained(cfg["save_dir"])
    tokenizer.save_pretrained(cfg["save_dir"])

    loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    _plot_loss(loss_history, cfg["save_dir"], cfg["model_id"])
    return loss_history

def _plot_loss(history, save_dir, name):
    if not history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(history, linewidth=2)
    plt.title(f"Loss por Step — {name}", fontsize=13)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.grid(True, alpha=0.4); plt.tight_layout()
    path = Path(save_dir) / "loss_curve.png"
    plt.savefig(path, dpi=120); plt.close()
    print(f"[INFO] Curva de loss: {path}")