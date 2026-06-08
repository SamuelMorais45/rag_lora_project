from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_EVAL   = ROOT / "data" / "evaluation"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DATASET_PATH = DATA_PROC / "dataset_gerado.jsonl"

MODEL_PATHS = {
    "causal-gpt2-medium" : MODELS_DIR / "lora_causal_model_1",
    "causal-opt-1.3b"    : MODELS_DIR / "lora_causal_model_2",
    "seq2seq-flan-t5"    : MODELS_DIR / "lora_seq2seq_model_1",
    "seq2seq-bart-large" : MODELS_DIR / "lora_seq2seq_model_2",
}

GENERATOR_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  
CHUNK_SIZE_DEFAULT = 500      
CHUNK_OVERLAP      = 50
MIN_PAIRS          = 100

LORA_CONFIGS = {
    "causal_1": {
        "model_id"      : "gpt2-medium",
        "save_dir"      : str(MODELS_DIR / "lora_causal_model_1"),
        "model_type"    : "causal",
        "target_modules": ["c_attn", "c_proj"],
        "lora_r"        : 16,
        "lora_alpha"    : 32,
        "lora_dropout"  : 0.1,
        "lr"            : 2e-4,
        "epochs"        : 3,
        "batch_size"    : 4,
    },
    "causal_2": {
        "model_id"      : "facebook/opt-1.3b",
        "save_dir"      : str(MODELS_DIR / "lora_causal_model_2"),
        "model_type"    : "causal",
        "target_modules": ["q_proj", "v_proj"],
        "lora_r"        : 8,
        "lora_alpha"    : 16,
        "lora_dropout"  : 0.05,
        "lr"            : 1e-4,
        "epochs"        : 3,
        "batch_size"    : 2,
    },
    "seq2seq_1": {
        "model_id"      : "google/flan-t5-base",
        "save_dir"      : str(MODELS_DIR / "lora_seq2seq_model_1"),
        "model_type"    : "seq2seq",
        "target_modules": ["q", "v"],
        "lora_r"        : 16,
        "lora_alpha"    : 32,
        "lora_dropout"  : 0.1,
        "lr"            : 3e-4,
        "epochs"        : 5,
        "batch_size"    : 8,
    },
    "seq2seq_2": {
        "model_id"      : "facebook/bart-large",
        "save_dir"      : str(MODELS_DIR / "lora_seq2seq_model_2"),
        "model_type"    : "seq2seq",
        "target_modules": ["q_proj", "v_proj"],
        "lora_r"        : 16,
        "lora_alpha"    : 32,
        "lora_dropout"  : 0.1,
        "lr"            : 2e-4,
        "epochs"        : 3,
        "batch_size"    : 4,
    },
}