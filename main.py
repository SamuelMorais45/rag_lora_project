# =============================================================================
# LABORATÓRIO: Clone do ChatGPT com FastAPI + Modelos HuggingFace (LoRA)
# =============================================================================
# Este arquivo é o coração da aplicação. Ele:
#   1. Carrega os 4 modelos fine-tunados com LoRA (2 causais + 2 seq2seq)
#   2. Expõe uma API REST via FastAPI
#   3. Serve o front-end estático (HTML/CSS/JS)
#   4. Processa mensagens do usuário e retorna respostas geradas pelos modelos
#
# Modelos treinados (confirmados via adapter_config.json):
#   causal_1  : EleutherAI/gpt-neo-125m  → models/lora_causal_model_1
#   causal_2  : facebook/opt-1.3b        → models/lora_causal_model_2
#   seq2seq_1 : google/mt5-small         → models/lora_seq2seq_model_1
#   seq2seq_2 : facebook/bart-large      → models/lora_seq2seq_model_2
#
# Como rodar (raiz do projeto, .venv ativa):
#   make api
#   -- ou --
#   .venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
# =============================================================================

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# INSTÂNCIA DA APLICAÇÃO FASTAPI
# =============================================================================
app = FastAPI(
    title="RAG + LoRA Chat — UFRN/CERES/DCT",
    description=(
        "API com 4 modelos fine-tunados via LoRA.\n"
        "Causais: GPT-Neo 125M, OPT-1.3B.\n"
        "Seq2Seq: MT5-Small, BART Large."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DICIONÁRIO GLOBAL DE MODELOS
# =============================================================================
# Chave  → ID do modelo (string usada em /modelos e /chat)
# Valor  → dicionário com "pipeline", "tipo" e "tokenizer"
MODELS: dict = {}

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INT = 0 if DEVICE == "cuda" else -1   # pipeline usa -1 para CPU

# =============================================================================
# METADADOS DOS MODELOS
# =============================================================================
# base_id  : ID exato do modelo base no HuggingFace (deve bater com
#            base_model_name_or_path do adapter_config.json)
# lora_dir : pasta onde os adaptadores LoRA foram salvos pelo treino
# tipo     : "causal" ou "seq2seq" (define qual pipeline e extração usar)
MODELOS_INFO = {
    "causal-gpt-neo-125m": {
        "id":        "causal-gpt-neo-125m",
        "nome":      "GPT-Neo 125M (LoRA)",
        "descricao": "Modelo causal EleutherAI GPT-Neo 125M fine-tunado com LoRA.",
        "tipo":      "causal",
        "base_id":   "EleutherAI/gpt-neo-125m",
        "lora_dir":  "models/lora_causal_model_1",
    },
    "causal-opt-1.3b": {
        "id":        "causal-opt-1.3b",
        "nome":      "OPT-1.3B (LoRA)",
        "descricao": "Modelo causal Meta OPT-1.3B fine-tunado com LoRA.",
        "tipo":      "causal",
        "base_id":   "facebook/opt-1.3b",
        "lora_dir":  "models/lora_causal_model_2",
    },
    "seq2seq-mt5-small": {
        "id":        "seq2seq-mt5-small",
        "nome":      "MT5-Small (LoRA)",
        "descricao": "Modelo encoder-decoder Google MT5-Small multilíngue fine-tunado com LoRA.",
        "tipo":      "seq2seq",
        "base_id":   "google/mt5-small",
        "lora_dir":  "models/lora_seq2seq_model_1",
    },
    "seq2seq-bart-large": {
        "id":        "seq2seq-bart-large",
        "nome":      "BART Large (LoRA)",
        "descricao": "Modelo encoder-decoder Meta BART Large fine-tunado com LoRA.",
        "tipo":      "seq2seq",
        "base_id":   "facebook/bart-large",
        "lora_dir":  "models/lora_seq2seq_model_2",
    },
}

# =============================================================================
# FUNÇÕES DE CARREGAMENTO
# =============================================================================

def carregar_modelo_causal(info: dict) -> dict | None:
    """
    Carrega um modelo causal (decoder-only) com adaptadores LoRA.

    Modelos causais geram texto continuando o prompt.
    O pipeline "text-generation" devolve prompt + resposta concatenados;
    por isso usamos return_full_text=False no /chat para receber
    apenas o texto novo gerado.

    Retorna dict com "pipeline", "tipo" e "tokenizer",
    ou None se a pasta lora_dir não existir (modelo ainda não treinado).
    """
    lora_dir = info["lora_dir"]
    if not Path(lora_dir).exists():
        logger.warning("  [SKIP] %s — pasta não encontrada: %s", info["id"], lora_dir)
        return None

    logger.info("  Carregando causal: %s + LoRA em %s", info["base_id"], lora_dir)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base  = AutoModelForCausalLM.from_pretrained(info["base_id"], torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, lora_dir)
    model = model.to(DEVICE)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE_INT,
    )

    logger.info("  ✓ %s carregado!", info["id"])
    return {"pipeline": pipe, "tipo": "causal", "tokenizer": tokenizer, "model": model}


def carregar_modelo_seq2seq(info: dict) -> dict | None:
    """
    Carrega um modelo seq2seq (encoder-decoder) com adaptadores LoRA.

    Diferença crucial em relação ao causal:
    O pipeline "text2text-generation" recebe só a instrução no encoder
    e o decoder gera a resposta limpa — sem o prompt concatenado.
    Por isso a extração no /chat é direta, sem cortar nada.

    Retorna dict com "pipeline", "tipo" e "tokenizer",
    ou None se a pasta lora_dir não existir.
    """
    lora_dir = info["lora_dir"]
    if not Path(lora_dir).exists():
        logger.warning("  [SKIP] %s — pasta não encontrada: %s", info["id"], lora_dir)
        return None

    logger.info("  Carregando seq2seq: %s + LoRA em %s", info["base_id"], lora_dir)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(lora_dir)

    base  = AutoModelForSeq2SeqLM.from_pretrained(info["base_id"], torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, lora_dir)
    model = model.to(DEVICE)
    model.eval()

    # text2text-generation foi removido no transformers >= 4.52.
    # Usamos text-generation com ForceTokensLogitsProcessor desativado.
    # A extração da resposta no /chat usa generate() direto, sem pipeline,
    # para modelos seq2seq — veja o endpoint /chat.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE_INT,
    )

    logger.info("  ✓ %s carregado!", info["id"])
    return {"pipeline": pipe, "tipo": "seq2seq", "tokenizer": tokenizer, "model": model}


# =============================================================================
# EVENTO DE INICIALIZAÇÃO
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Carrega todos os 4 modelos quando o servidor sobe.
    Modelos com pasta inexistente são pulados sem derrubar a API —
    útil se ainda estiver treinando algum modelo.
    """
    global MODELS
    logger.info("=" * 60)
    logger.info("  INICIANDO SERVIDOR — Carregando modelos LoRA...")
    logger.info("  Dispositivo: %s", DEVICE.upper())
    logger.info("=" * 60)

    for model_id, info in MODELOS_INFO.items():
        if info["tipo"] == "causal":
            resultado = carregar_modelo_causal(info)
        else:
            resultado = carregar_modelo_seq2seq(info)

        if resultado:
            MODELS[model_id] = resultado

    logger.info("=" * 60)
    logger.info(
        "  ✓ %d/4 modelo(s) carregado(s): %s",
        len(MODELS), list(MODELS.keys())
    )
    logger.info("=" * 60)


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class ChatRequest(BaseModel):
    """
    Schema da requisição — usa os mesmos nomes do professor
    (modelo, mensagem, max_tokens, temperatura) para que o
    index.html original funcione sem alteração.
    """
    modelo:      str
    mensagem:    str
    max_tokens:  Optional[int]   = 200
    temperatura: Optional[float] = 0.7


class ChatResponse(BaseModel):
    resposta:       str
    modelo:         str
    tokens_gerados: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/modelos", response_class=JSONResponse)
async def listar_modelos():
    """
    GET /modelos

    Retorna apenas os modelos carregados com sucesso.
    O front-end usa este endpoint para popular o dropdown de seleção.
    """
    disponiveis = [
        {
            "id":        info["id"],
            "nome":      info["nome"],
            "descricao": info["descricao"],
        }
        for model_id, info in MODELOS_INFO.items()
        if model_id in MODELS
    ]
    return {"modelos": disponiveis}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat

    Recebe a mensagem do usuário, gera resposta com o modelo
    selecionado e devolve o texto gerado.

    Extração da resposta por tipo:
    - Causal  : return_full_text=False → pipeline devolve só o texto novo.
    - Seq2seq : resultado[0]["generated_text"] já é a resposta limpa.
    """
    if request.modelo not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{request.modelo}' não encontrado. "
                f"Disponíveis: {list(MODELS.keys())}"
            )
        )

    if not request.mensagem.strip():
        raise HTTPException(status_code=400, detail="A mensagem não pode ser vazia.")

    logger.info(
        "[CHAT] modelo='%s' | mensagem='%s...'",
        request.modelo, request.mensagem[:50]
    )

    entrada   = MODELS[request.modelo]
    pipe      = entrada["pipeline"]
    tipo      = entrada["tipo"]
    tokenizer = entrada["tokenizer"]

    try:
        if tipo == "causal":
            # ── Modelos causais (GPT-Neo, OPT) ──────────────────────────
            # Prompt no mesmo formato usado durante o treino (build_prompt)
            prompt = f"### Instruction:\n{request.mensagem}\n\n### Response:\n"
            resultado = pipe(
                prompt,
                max_new_tokens        = request.max_tokens,
                temperature           = request.temperatura,
                do_sample             = True,
                top_p                 = 0.9,
                repetition_penalty    = 1.3,
                no_repeat_ngram_size  = 4,
                pad_token_id          = tokenizer.eos_token_id,
                eos_token_id          = tokenizer.eos_token_id,
                num_return_sequences  = 1,
                return_full_text      = False,
            )
            resposta = resultado[0]["generated_text"].strip()

        else:
            # ── Modelos seq2seq (MT5, BART) ──────────────────────────────
            # text2text-generation foi removido no transformers >= 4.52.
            # Usamos generate() diretamente: tokenizamos a entrada,
            # geramos com o decoder e decodificamos a saída.
            model  = entrada["model"]
            inputs = tokenizer(
                request.mensagem,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 512,
            ).to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens      = request.max_tokens,
                    no_repeat_ngram_size= 3,
                    early_stopping      = True,
                    num_beams           = 4,
                )
            resposta = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        if not resposta:
            resposta = "[O modelo não gerou texto. Tente aumentar max_tokens.]"

        tokens_gerados = len(tokenizer.encode(resposta))
        logger.info("  ✓ %d tokens gerados", tokens_gerados)

        return ChatResponse(
            resposta       = resposta,
            modelo         = request.modelo,
            tokens_gerados = tokens_gerados,
        )

    except Exception as e:
        logger.error("Erro na geração: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar resposta: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    GET /health

    Verifica quais modelos estão carregados e o dispositivo em uso.
    """
    return {
        "status":             "ok" if MODELS else "sem_modelos",
        "modelos_carregados": list(MODELS.keys()),
        "quantidade":         len(MODELS),
        "device":             DEVICE,
    }


# =============================================================================
# SERVIR O FRONT-END
# =============================================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a página principal do chat (static/index.html)."""
    html_path = os.path.join("static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
