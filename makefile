ifeq ($(OS),Windows_NT)
    PYTHON        := python
    VENV_ACTIVATE := .venv/Scripts/activate
    VENV_PYTHON   := .venv/Scripts/python.exe
    VENV_PIP      := .venv/Scripts/pip.exe
    RM_CACHE      := for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    RM_DIR        := rd /s /q
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

.PHONY: setup install dirs stage1 stage2 stage3 stage3-nb api api-bg stop-api \
        clean clean-models clean-data status all help


help:
	@echo ""
	@echo "  RAG + LoRA Project - Comandos disponiveis"
	@echo "  make setup          Cria a .venv (rode uma unica vez)"
	@echo "  make install        Instala requirements.txt na .venv"
	@echo "  make dirs           Cria todas as pastas necessarias"
	@echo "  make stage1         Etapa 1: executa 01_rag.ipynb -> gera dataset"
	@echo "  make stage2         Etapa 2: executa 02_lora.ipynb -> treina modelos"
	@echo "  make stage3         Etapa 3: avalia os 4 modelos (CLI, sem notebook)"
	@echo "  make stage3-nb      Etapa 3: avalia via notebook (com graficos inline)"
	@echo "  make api            Etapa 4: sobe FastAPI em primeiro plano"
	@echo "  make api-bg         Etapa 4: sobe FastAPI em background (porta $(API_PORT))"
	@echo "  make all            install + dirs + stage1 + stage2 + stage3"
	@echo "  make status         Mostra o que ja foi gerado"
	@echo "  make clean          Remove __pycache__ e arquivos temporarios"
	@echo "  make clean-models   Remove os modelos treinados (para re-treinar)"
	@echo "  make clean-data     Remove o dataset gerado (para regenerar)"
	@echo ""


setup:
	@echo ">>> Criando .venv..."
	$(PYTHON) -m venv .venv
	@echo ""
	@echo "  $(SETUP_MSG)"
	@echo "  Depois execute: make install"
	@echo ""


install:
	@echo ">>> Instalando dependencias..."
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo ">>> Baixando tokenizador NLTK (necessario para metricas)..."
	$(VENV_PYTHON) -c "import nltk; nltk.download('punkt_tab', quiet=True)"
	@echo ">>> Dependencias instaladas com sucesso!"

dirs:
	@echo ">>> Criando estrutura de pastas..."
ifeq ($(OS),Windows_NT)
	@if not exist data                        md data
	@if not exist data\raw                    md data\raw
	@if not exist data\processed              md data\processed
	@if not exist data\evaluation             md data\evaluation
	@if not exist models                      md models
	@if not exist models\lora_causal_model_1  md models\lora_causal_model_1
	@if not exist models\lora_causal_model_2  md models\lora_causal_model_2
	@if not exist models\lora_seq2seq_model_1 md models\lora_seq2seq_model_1
	@if not exist models\lora_seq2seq_model_2 md models\lora_seq2seq_model_2
	@if not exist reports                     md reports
	@if not exist static                      md static
else
	$(MKDIR) $(DATA_RAW) $(DATA_PROC) $(DATA_EVAL)
	$(MKDIR) $(MODELS_DIR)/lora_causal_model_1
	$(MKDIR) $(MODELS_DIR)/lora_causal_model_2
	$(MKDIR) $(MODELS_DIR)/lora_seq2seq_model_1
	$(MKDIR) $(MODELS_DIR)/lora_seq2seq_model_2
	$(MKDIR) $(REPORTS_DIR) $(STATIC_DIR)
endif
	@echo ">>> Pastas criadas."
	@echo ">>> ATENCAO: coloque o PDF-fonte em data/raw antes de make stage1"


stage1:
	@echo ">>> [Etapa 1] Gerando dataset (sem notebook)..."
	@echo "    (pode levar alguns minutos - modelo gerador precisa de GPU)"
	$(VENV_PYTHON) -m src.dataset_generation.run_stage1
	@echo ">>> Dataset gerado em: $(DATASET_FILE)"
	@echo ""
	@echo "    Opcoes uteis:"
	@echo "    --pdf data/raw/arquivo.pdf   (escolher PDF especifico)"
	@echo "    --chunk-size 300             (testar tamanho de chunk)"
	@echo ""


stage2:
	@echo ">>> [Etapa 2] Treinando os 4 modelos com LoRA (sem notebook)..."
	@echo "    (pode levar horas em CPU - recomendado rodar com GPU)"
	$(VENV_PYTHON) -m src.training.run_stage2
	@echo ">>> Modelos salvos em $(MODELS_DIR)/"
	@echo ""
	@echo "    Para treinar so um modelo:"
	@echo "    $(VENV_PYTHON) -m src.training.run_stage2 --model causal_1"
	@echo "    Opcoes: causal_1  causal_2  seq2seq_1  seq2seq_2"
	@echo ""


stage3:
	@echo ">>> [Etapa 3] Avaliando os 4 modelos..."
	$(VENV_PYTHON) -m src.evaluation.evaluate_models
	@echo ">>> Resultados em $(DATA_EVAL)/resultados_avaliacao.csv"
	@echo ">>> Grafico radar em $(REPORTS_DIR)/radar_comparativo.png"


stage3-nb:
	@echo ">>> [Etapa 3] Avaliando via notebook..."
	$(VENV_PYTHON) -m jupyter nbconvert \
	    --to notebook \
	    --execute \
	    --inplace \
	    --ExecutePreprocessor.timeout=7200 \
	    $(NB_EVAL)
	@echo ">>> Notebook executado com saidas salvas."


api:
	@echo ">>> [Etapa 4] Subindo API em http://localhost:$(API_PORT)"
	@echo "    Swagger:  http://localhost:$(API_PORT)/docs"
	@echo "    Frontend: http://localhost:$(API_PORT)"
	@echo "    Pressione Ctrl+C para encerrar."
	$(VENV_PYTHON) -m uvicorn main:app \
	    --reload \
	    --host $(API_HOST) \
	    --port $(API_PORT)


api-bg:
ifeq ($(OS),Windows_NT)
	@echo "api-bg nao suportado no Windows."
	@echo "Abra um segundo terminal e use: make api"
else
	$(VENV_PYTHON) -m uvicorn main:app \
	    --host $(API_HOST) \
	    --port $(API_PORT) \
	    --log-level info \
	    > api.log 2>&1 & echo $$! > .api.pid
	@echo ">>> API em background. PID: $$(cat .api.pid)"
	@echo "    Logs:     tail -f api.log"
	@echo "    Encerrar: make stop-api"
endif

stop-api:
ifeq ($(OS),Windows_NT)
	@echo "Use o Gerenciador de Tarefas para encerrar o processo uvicorn."
else
	@if [ -f .api.pid ]; then \
	    kill $$(cat .api.pid) && rm .api.pid; \
	    echo ">>> API encerrada."; \
	else \
	    echo ">>> Nenhuma API rodando em background."; \
	fi
endif


status:
ifeq ($(OS),Windows_NT)
	@echo.
	@echo   -- Status do projeto --
	@if exist data\processed\dataset_gerado.jsonl             (echo   [OK]   Dataset gerado) else (echo   [FALTA] Dataset - rode: make stage1)
	@if exist models\lora_causal_model_1\adapter_config.json  (echo   [OK]   Modelo causal 1: GPT-2 Medium)  else (echo   [FALTA] Modelo causal 1 - rode: make stage2)
	@if exist models\lora_causal_model_2\adapter_config.json  (echo   [OK]   Modelo causal 2: OPT-1.3B)      else (echo   [FALTA] Modelo causal 2 - rode: make stage2)
	@if exist models\lora_seq2seq_model_1\adapter_config.json (echo   [OK]   Modelo seq2seq 1: Flan-T5)      else (echo   [FALTA] Modelo seq2seq 1 - rode: make stage2)
	@if exist models\lora_seq2seq_model_2\adapter_config.json (echo   [OK]   Modelo seq2seq 2: BART Large)   else (echo   [FALTA] Modelo seq2seq 2 - rode: make stage2)
	@if exist data\evaluation\resultados_avaliacao.csv        (echo   [OK]   Avaliacao concluida)             else (echo   [FALTA] Avaliacao - rode: make stage3)
	@if exist reports\radar_comparativo.png                   (echo   [OK]   Grafico radar gerado)            else (echo   [FALTA] Grafico - rode: make stage3)
	@echo.
else
	@echo ""
	@echo "  -- Status do projeto --"
	@test -f $(DATASET_FILE) \
	    && echo "  [OK]    Dataset: $(DATASET_FILE)" \
	    || echo "  [FALTA] Dataset - rode: make stage1"
	@test -f $(MODELS_DIR)/lora_causal_model_1/adapter_config.json \
	    && echo "  [OK]    Modelo causal 1 : GPT-2 Medium" \
	    || echo "  [FALTA] Modelo causal 1 - rode: make stage2"
	@test -f $(MODELS_DIR)/lora_causal_model_2/adapter_config.json \
	    && echo "  [OK]    Modelo causal 2 : OPT-1.3B" \
	    || echo "  [FALTA] Modelo causal 2 - rode: make stage2"
	@test -f $(MODELS_DIR)/lora_seq2seq_model_1/adapter_config.json \
	    && echo "  [OK]    Modelo seq2seq 1: Flan-T5" \
	    || echo "  [FALTA] Modelo seq2seq 1 - rode: make stage2"
	@test -f $(MODELS_DIR)/lora_seq2seq_model_2/adapter_config.json \
	    && echo "  [OK]    Modelo seq2seq 2: BART Large" \
	    || echo "  [FALTA] Modelo seq2seq 2 - rode: make stage2"
	@test -f $(DATA_EVAL)/resultados_avaliacao.csv \
	    && echo "  [OK]    Avaliacao: $(DATA_EVAL)/resultados_avaliacao.csv" \
	    || echo "  [FALTA] Avaliacao - rode: make stage3"
	@test -f $(REPORTS_DIR)/radar_comparativo.png \
	    && echo "  [OK]    Grafico radar: $(REPORTS_DIR)/radar_comparativo.png" \
	    || echo "  [FALTA] Grafico - rode: make stage3"
	@echo ""
endif


all: install dirs stage1 stage2 stage3
	@echo ""
	@echo "  Pipeline completo finalizado!"
	@echo "  Agora execute 'make api' para subir a API."
	@echo ""


clean:
	@echo ">>> Limpando __pycache__ e arquivos temporarios..."
	$(RM_CACHE)
ifeq ($(OS),Windows_NT)
	@if exist .pytest_cache  rd /s /q .pytest_cache
	@if exist api.log        del /q api.log
else
	$(RM_DIR) .pytest_cache 2>/dev/null || true
	$(RM_DIR) api.log 2>/dev/null || true
endif
	@echo ">>> Limpeza concluida."

clean-models:
	@echo ">>> Removendo modelos treinados..."
ifeq ($(OS),Windows_NT)
	@if exist models\lora_causal_model_1   rd /s /q models\lora_causal_model_1
	@if exist models\lora_causal_model_2   rd /s /q models\lora_causal_model_2
	@if exist models\lora_seq2seq_model_1  rd /s /q models\lora_seq2seq_model_1
	@if exist models\lora_seq2seq_model_2  rd /s /q models\lora_seq2seq_model_2
else
	$(RM_DIR) $(MODELS_DIR)/lora_causal_model_1
	$(RM_DIR) $(MODELS_DIR)/lora_causal_model_2
	$(RM_DIR) $(MODELS_DIR)/lora_seq2seq_model_1
	$(RM_DIR) $(MODELS_DIR)/lora_seq2seq_model_2
endif
	@echo ">>> Modelos removidos. Execute 'make stage2' para re-treinar."

clean-data:
	@echo ">>> Removendo dataset gerado..."
ifeq ($(OS),Windows_NT)
	@if exist data\processed  rd /s /q data\processed
else
	$(RM_DIR) $(DATA_PROC)
endif
	@echo ">>> Dataset removido. Execute 'make stage1' para regenerar."