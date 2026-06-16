import argparse
import logging

from src.training.trainer import train_model
from src.utils.config import DATASET_PATH, LORA_CONFIGS
from src.utils.helpers import load_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Etapa 2: fine-tuning com LoRA")
    parser.add_argument(
        "--model", nargs="+", dest="models",
        choices=list(LORA_CONFIGS.keys()),
        default=list(LORA_CONFIGS.keys()),
        help="Modelos a treinar (padrao: todos os 4)",
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        logger.error(
            "Dataset nao encontrado: %s\n"
            "Rode 'make stage1' primeiro.",
            DATASET_PATH,
        )
        return

    samples = load_jsonl(str(DATASET_PATH))
    logger.info("Dataset: %d amostras", len(samples))

    for key in args.models:
        cfg = LORA_CONFIGS[key]
        logger.info("=" * 60)
        logger.info("Treinando: %s (%s)", cfg["model_id"], cfg["model_type"])
        logger.info("=" * 60)
        loss = train_model(cfg, samples)
        logger.info(
            "Concluido: %s | loss final=%.4f | salvo em: %s",
            cfg["model_id"],
            loss[-1] if loss else float("nan"),
            cfg["save_dir"],
        )

    logger.info("Todos os modelos selecionados foram treinados.")


if __name__ == "__main__":
    main()
