import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from src.dataset_generation.chunking import chunk_text
from src.dataset_generation.curate_dataset import is_valid
from src.dataset_generation.extract_pdf import extract_text
from src.dataset_generation.generate_qa import generate_pair, load_generator
from src.utils.config import (
    CHUNK_OVERLAP, CHUNK_SIZE_DEFAULT, DATA_PROC, DATA_RAW,
    DATASET_PATH, GENERATOR_MODEL_ID, MIN_PAIRS,
)
from src.utils.helpers import save_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_pdf(pdf_arg: str | None) -> Path:
    if pdf_arg:
        p = Path(pdf_arg)
        if not p.exists():
            raise FileNotFoundError(f"PDF nao encontrado: {p}")
        return p
    pdfs = list(DATA_RAW.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Nenhum PDF em {DATA_RAW}. "
            f"Coloque o PDF-fonte nessa pasta antes de rodar."
        )
    if len(pdfs) > 1:
        logger.warning("Multiplos PDFs encontrados. Usando: %s", pdfs[0])
    return pdfs[0]


def run(pdf_path: Path, chunk_sizes: list[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Dispositivo : %s", device)
    logger.info("PDF-fonte : %s", pdf_path)

    logger.info("Extraindo texto do PDF...")
    text = extract_text(str(pdf_path))
    logger.info("Texto extraido: %d caracteres", len(text))

    logger.info("Carregando modelo gerador: %s", GENERATOR_MODEL_ID)
    model, tokenizer = load_generator(GENERATOR_MODEL_ID, device)

    all_pairs: list[dict] = []

    for chunk_size in chunk_sizes:
        logger.info("--- chunk_size=%d ---", chunk_size)
        chunks = chunk_text(text, max_length=chunk_size, overlap=CHUNK_OVERLAP)
        logger.info("Chunks: %d", len(chunks))

        pairs: list[dict] = []
        for chunk in tqdm(chunks, desc=f" Gerando pares (chunk={chunk_size})"):
            pair = generate_pair(model, tokenizer, chunk, device)
            if pair and is_valid(pair):
                pair["_chunk_size"] = chunk_size
                pairs.append(pair)

        removed = len(chunks) - len(pairs)
        logger.info(
            "chunk_size=%d | validos=%d | removidos=%d (%.1f%%)",
            chunk_size, len(pairs), removed,
            (removed / len(chunks) * 100) if chunks else 0,
        )
        all_pairs.extend(pairs)

    # Deduplicacao por Instruction
    seen, unique = set(), []
    for p in all_pairs:
        key = p["Instruction"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append({k: v for k, v in p.items() if k != "_chunk_size"})

    logger.info("Pares unicos apos deduplicacao: %d", len(unique))

    if len(unique) < MIN_PAIRS:
        logger.warning(
            "Dataset com %d pares (minimo: %d). "
            "Tente um PDF maior ou chunk menor.",
            len(unique), MIN_PAIRS,
        )

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    save_jsonl(unique, str(DATASET_PATH))
    logger.info("Dataset salvo: %s (%d pares)", DATASET_PATH, len(unique))


def main():
    parser = argparse.ArgumentParser(description="Etapa 1: geracao do dataset")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Caminho do PDF (padrao: primeiro PDF em data/raw/)")
    parser.add_argument("--chunk-size", type=int, action="append",
                        dest="chunk_sizes",
                        help="Tamanho do chunk em caracteres (repita para comparar dois valores)")
    args = parser.parse_args()

    chunk_sizes = args.chunk_sizes or [CHUNK_SIZE_DEFAULT]
    pdf_path = find_pdf(args.pdf)
    run(pdf_path, chunk_sizes)


if __name__ == "__main__":
    main()
