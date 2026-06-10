ifeq ($(OS),Windows_NT)
    PYTHON       := python
    VENV_ACTIVATE := .venv\Scripts\activate
    VENV_PYTHON   := .venv\Scripts\python.exe
    VENV_PIP      := .venv\Scripts\pip.exe
    RM_CACHE      := for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    RM_DIR        := rd /s /q
    MKDIR         := mkdir
    SEP           := \\
    SETUP_MSG     := Ambiente criado. Ative com: .venv\Scripts\activate
else
    PYTHON        := python3
    VENV_ACTIVATE := source .venv/bin/activate
    VENV_PYTHON   := .venv/bin/python
    VENV_PIP      := .venv/bin/pip
    RM_CACHE      := find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    RM_DIR        := rm -rf
    MKDIR         := mkdir -p
    SEP           := /
    SETUP_MSG     := Ambiente criado. Ative com: source .venv/bin/activate
endif
 
NOTEBOOKS_DIR  := notebooks
DATA_RAW       := data/raw
DATA_PROC      := data/processed
DATA_EVAL      := data/evaluation
MODELS_DIR     := models
REPORTS_DIR    := reports
STATIC_DIR     := static
DATASET_FILE   := $(DATA_PROC)/dataset_gerado.jsonl
 
NB_RAG         := $(NOTEBOOKS_DIR)/01_rag.ipynb
NB_LORA        := $(NOTEBOOKS_DIR)/02_lora.ipynb
NB_EVAL        := $(NOTEBOOKS_DIR)/03_avaliacao_modelo_finetuned.ipynb
 
API_HOST       := 0.0.0.0
API_PORT       := 8000
 
.PHONY: setup install dirs stage1 stage2 stage3 api api-bg \
        clean clean-models clean-data status all help
 
 
help:
	@echo ""
	@echo "  RAG + LoRA Project — Comandos disponíveis"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make setup          Cria a .venv (rode uma única vez)"
	@echo "  make install        Instala requirements.txt na .venv ativa"
	@echo "  make dirs           Cria todas as pastas necessárias"
	@echo "  make stage1         Etapa 1: executa 01_rag.ipynb → gera dataset"
	@echo "  make stage2         Etapa 2: executa 02_lora.ipynb → treina modelos"
	@echo "  make stage3         Etapa 3: avalia os 4 modelos (CLI, sem notebook)"
	@echo "  make stage3-nb      Etapa 3: avalia via notebook (com gráficos inline)"
	@echo "  make api            Etapa 4: sobe FastAPI em primeiro plano"
	@echo "  make api-bg         Etapa 4: sobe FastAPI em background (porta $(API_PORT))"
	@echo "  make all            install + dirs + stage1 + stage2 + stage3"
	@echo "  make status         Mostra o que já foi gerado"
	@echo "  make clean          Remove __pycache__ e arquivos temporários"
	@echo "  make clean-models   Remove os modelos treinados (para re-treinar)"
	@echo "  make clean-data     Remove o dataset gerado (para regenerar)"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo ""
 
 
setup:
	@echo ">>> Criando .venv..."
	$(PYTHON) -m venv .venv
	@echo ""
	@echo "  $(SETUP_MSG)"
	@echo "  Depois execute: make install"
	@echo ""
 
 
install:
	@echo ">>> Instalando dependências..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo ">>> Baixando tokenizador NLTK (necessário para métricas)..."
	$(VENV_PYTHON) -c "import nltk; nltk.download('punkt_tab', quiet=True)"
	@echo ">>> Dependências instaladas com sucesso!"
 

dirs:
	@echo ">>> Criando estrutura de pastas..."
	$(MKDIR) $(DATA_RAW)
	$(MKDIR) $(DATA_PROC)
	$(MKDIR) $(DATA_EVAL)
	$(MKDIR) $(MODELS_DIR)/lora_causal_model_1
	$(MKDIR) $(MODELS_DIR)/lora_causal_model_2
	$(MKDIR) $(MODELS_DIR)/lora_seq2seq_model_1
	$(MKDIR) $(MODELS_DIR)/lora_seq2seq_model_2
	$(MKDIR) $(REPORTS_DIR)
	$(MKDIR) $(STATIC_DIR)
	@echo ">>> Pastas criadas."
	@echo ">>> ATENÇÃO: coloque o PDF-fonte em $(DATA_RAW)/ antes de 'make stage1'."
 

stage1: $(NB_RAG)
	@echo ">>> [Etapa 1] Gerando dataset via 01_rag.ipynb..."
	@echo "    (pode levar alguns minutos — o modelo gerador precisa de GPU)"
	$(VENV_PYTHON) -m jupyter nbconvert \
	    --to notebook \
	    --execute \
	    --inplace \
	    --ExecutePreprocessor.timeout=3600 \
	    $(NB_RAG)
	@echo ">>> Dataset gerado em: $(DATASET_FILE)"
 

stage2: $(DATASET_FILE) $(NB_LORA)
	@echo ">>> [Etapa 2] Treinando os 4 modelos com LoRA via 02_lora.ipynb..."
	@echo "    (pode levar horas em CPU — recomendado rodar com GPU)"
	$(VENV_PYTHON) -m jupyter nbconvert \
	    --to notebook \
	    --execute \
	    --inplace \
	    --ExecutePreprocessor.timeout=86400 \
	    $(NB_LORA)
	@echo ">>> Modelos salvos em $(MODELS_DIR)/"
 
 
stage3: $(DATASET_FILE)
	@echo ">>> [Etapa 3] Avaliando os 4 modelos..."
	$(VENV_PYTHON) -m src.evaluation.evaluate_models
	@echo ">>> Resultados em $(DATA_EVAL)/resultados_avaliacao.csv"
	@echo ">>> Gráfico radar em $(REPORTS_DIR)/radar_comparativo.png"
 
 
stage3-nb: $(DATASET_FILE) $(NB_EVAL)
	@echo ">>> [Etapa 3] Avaliando via notebook..."
	$(VENV_PYTHON) -m jupyter nbconvert \
	    --to notebook \
	    --execute \
	    --inplace \
	    --ExecutePreprocessor.timeout=7200 \
	    $(NB_EVAL)
	@echo ">>> Notebook executado com saídas salvas."
 
 
api:
	@echo ">>> [Etapa 4] Subindo API em http://$(API_HOST):$(API_PORT)"
	@echo "    Swagger: http://localhost:$(API_PORT)/docs"
	@echo "    Frontend: http://localhost:$(API_PORT)"
	@echo "    Pressione Ctrl+C para encerrar."
	$(VENV_PYTHON) -m uvicorn src.api.main:app \
	    --reload \
	    --host $(API_HOST) \
	    --port $(API_PORT)
 

api-bg:
	@echo ">>> Subindo API em background na porta $(API_PORT)..."
	$(VENV_PYTHON) -m uvicorn src.api.main:app \
	    --host $(API_HOST) \
	    --port $(API_PORT) \
	    --log-level info \
	    > api.log 2>&1 & echo $$! > .api.pid
	@echo "    PID: $$(cat .api.pid)"
	@echo "    Logs: tail -f api.log"
	@echo "    Para encerrar: make stop-api"
 
stop-api:
	@if [ -f .api.pid ]; then \
	    kill $$(cat .api.pid) && rm .api.pid; \
	    echo ">>> API encerrada."; \
	else \
	    echo ">>> Nenhuma API rodando em background."; \
	fi
 
 
status:
	@echo ""
	@echo "  ── Status do projeto ──────────────────────────────────"
	@echo -n "  Dataset (stage1):  " && \
	    (test -f $(DATASET_FILE) && echo "✅  $(DATASET_FILE)" || echo "❌  Não gerado ainda")
	@echo -n "  Modelo causal 1 :  " && \
	    (test -d $(MODELS_DIR)/lora_causal_model_1/adapter_config.json \
	     || test -f $(MODELS_DIR)/lora_causal_model_1/adapter_config.json \
	     && echo "✅  GPT-2 Medium" || echo "❌  Não treinado")
	@echo -n "  Modelo causal 2 :  " && \
	    (test -f $(MODELS_DIR)/lora_causal_model_2/adapter_config.json \
	     && echo "✅  OPT-1.3B" || echo "❌  Não treinado")
	@echo -n "  Modelo seq2seq 1:  " && \
	    (test -f $(MODELS_DIR)/lora_seq2seq_model_1/adapter_config.json \
	     && echo "✅  Flan-T5" || echo "❌  Não treinado")
	@echo -n "  Modelo seq2seq 2:  " && \
	    (test -f $(MODELS_DIR)/lora_seq2seq_model_2/adapter_config.json \
	     && echo "✅  BART Large" || echo "❌  Não treinado")
	@echo -n "  Resultados eval :  " && \
	    (test -f $(DATA_EVAL)/resultados_avaliacao.csv \
	     && echo "✅  $(DATA_EVAL)/resultados_avaliacao.csv" || echo "❌  Não avaliado")
	@echo -n "  Gráfico radar   :  " && \
	    (test -f $(REPORTS_DIR)/radar_comparativo.png \
	     && echo "✅  $(REPORTS_DIR)/radar_comparativo.png" || echo "❌  Não gerado")
	@echo "  ───────────────────────────────────────────────────────"
	@echo ""
 
 
all: install dirs stage1 stage2 stage3
	@echo ""
	@echo "  Pipeline completo finalizado!"
	@echo "  Agora execute 'make api' para subir a API."
	@echo ""
 
 
clean:
	@echo ">>> Limpando __pycache__ e arquivos temporários..."
	$(RM_CACHE)
	$(RM_DIR) .pytest_cache 2>/dev/null || true
	$(RM_DIR) api.log 2>/dev/null || true
	@echo ">>> Limpeza concluída."
 
clean-models:
	@echo ">>> Removendo modelos treinados..."
	$(RM_DIR) $(MODELS_DIR)/lora_causal_model_1
	$(RM_DIR) $(MODELS_DIR)/lora_causal_model_2
	$(RM_DIR) $(MODELS_DIR)/lora_seq2seq_model_1
	$(RM_DIR) $(MODELS_DIR)/lora_seq2seq_model_2
	@echo ">>> Modelos removidos. Execute 'make stage2' para re-treinar."
 
clean-data:
	@echo ">>> Removendo dataset gerado..."
	$(RM_DIR) $(DATA_PROC)
	@echo ">>> Dataset removido. Execute 'make stage1' para regenerar."