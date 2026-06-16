import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from tqdm import tqdm

from src.evaluation.bleu import compute_bleu
from src.evaluation.faithfulness import compute_faithfulness
from src.evaluation.metrics import compute_ppl, generate_response
from src.evaluation.plan_adherence import compute_plan_adherence
from src.evaluation.relevance import compute_relevance
from src.evaluation.rouge import compute_rouge
from src.utils.config import DATA_EVAL, DATASET_PATH, LORA_CONFIGS, REPORTS_DIR
from src.utils.helpers import load_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES: int | None = None



def load_model_for_eval(cfg: dict):
    lora_path = cfg["save_dir"]
    if not Path(lora_path).exists():
        logger.warning("  Caminho não encontrado: %s — pulando modelo.", lora_path)
        return None, None

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_type"] == "causal":
        base  = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=dtype)
    else:
        base  = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_id"], torch_dtype=dtype)

    model = PeftModel.from_pretrained(base, lora_path)
    model = model.to(DEVICE).eval()
    return model, tokenizer


def evaluate_single(cfg: dict, samples: list[dict]) -> dict | None:
    label = cfg["model_id"]
    logger.info("\n%s\n[EVAL] Avaliando: %s\n%s", "="*60, label, "="*60)

    model, tokenizer = load_model_for_eval(cfg)
    if model is None:
        return None

    model_type = cfg["model_type"]

    generated = []
    logger.info("Gerando respostas para %d amostras...", len(samples))
    for s in tqdm(samples, desc=f"  {label[:30]}"):
        resp = generate_response(model, tokenizer, s["Instruction"], model_type)
        generated.append(resp)

    references = [s["Output"] for s in samples]
    hypotheses = generated

    ppl_scores = [compute_ppl(model, tokenizer, s, model_type) for s in tqdm(samples, desc="  PPL")]
    mean_ppl   = float(np.mean(ppl_scores))

    bleu_summary, _ = compute_bleu(hypotheses, references)

    df_rouge    = compute_rouge(hypotheses, references)
    mean_rouge1 = float(df_rouge["ROUGE-1 F1"].mean())
    mean_rouge2 = float(df_rouge["ROUGE-2 F1"].mean())
    mean_rougeL = float(df_rouge["ROUGE-L F1"].mean())

    faith_scores   = compute_faithfulness(samples, generated)
    relev_scores   = compute_relevance(samples, generated)
    plan_scores    = compute_plan_adherence(samples, generated)

    result = {
        "modelo"          : label,
        "PPL"             : round(mean_ppl, 2),
        "BLEU"            : round(bleu_summary["bleu_corpus"], 2),
        "ROUGE-1"         : round(mean_rouge1, 3),
        "ROUGE-2"         : round(mean_rouge2, 3),
        "ROUGE-L"         : round(mean_rougeL, 3),
        "Faithfulness"    : round(float(np.mean(faith_scores)), 3),
        "Answer Relevance": round(float(np.mean(relev_scores)), 3),
        "Plan Adherence"  : round(float(np.mean(plan_scores)),  3),
    }

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


def plot_radar(df: pd.DataFrame, out_path: Path):
    METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Faithfulness", "Answer Relevance", "Plan Adherence"]
    N = len(METRICS)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    colors  = plt.cm.Set2.colors

    for i, (_, row) in enumerate(df.iterrows()):
        vals   = [row[m] for m in METRICS] + [row[METRICS[0]]]
        label  = row["modelo"].split("/")[-1]  
        ax.plot(angles, vals, "o-", linewidth=2, color=colors[i % len(colors)], label=label)
        ax.fill(angles, vals, alpha=0.1,         color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Comparação dos 4 Modelos — Gráfico Radar", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Radar salvo em %s", out_path)


def main(filter_model: str | None = None):
    if not DATASET_PATH.exists():
        logger.error("Dataset não encontrado: %s", DATASET_PATH)
        logger.error("Execute a Etapa 1 (01_rag.ipynb) antes de avaliar.")
        return

    samples = load_jsonl(str(DATASET_PATH))
    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]
    logger.info("Dataset: %d amostras de %s", len(samples), DATASET_PATH)

    results = []
    for key, cfg in LORA_CONFIGS.items():
        if filter_model and cfg["model_id"] not in filter_model and key not in filter_model:
            continue
        res = evaluate_single(cfg, samples)
        if res:
            results.append(res)

    if not results:
        logger.warning("Nenhum modelo avaliado. Verifique se os modelos foram treinados.")
        return

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("            TABELA COMPARATIVA — TODOS OS MODELOS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    best_idx   = df["ROUGE-L"].idxmax()
    best_model = df.loc[best_idx, "modelo"]
    logger.info("🏆 Melhor modelo por ROUGE-L: %s (%.3f)", best_model, df.loc[best_idx, "ROUGE-L"])

    DATA_EVAL.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_EVAL / "resultados_avaliacao.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(" CSV salvo em %s", csv_path)

    plot_radar(df, REPORTS_DIR / "radar_comparativo.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia os modelos fine-tunados.")
    parser.add_argument("--model", type=str, default=None,
                        help="ID do modelo a avaliar (padrão: todos)")
    args = parser.parse_args()
    main(filter_model=args.model)
