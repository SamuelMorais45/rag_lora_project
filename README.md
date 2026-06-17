# RAG + LoRA Fine-tuning — Glossário Aeronáutico ANAC

Projeto final da disciplina **Tópicos Avançados em IA A** — UFRN/CERES/DCT  
Prof. Dr. Thommas K. S. Flores — 2026.1

Pipeline completo de geração de dataset via RAG, fine-tuning com LoRA em quatro modelos Transformer e exposição dos modelos via API FastAPI com frontend de chat.

---

## Visão geral

```
PDF (glossário ANAC)
  └── Etapa 1 — RAG      →  data/processed/dataset_gerado.jsonl  (≥ 100 pares)
        └── Etapa 2 — LoRA   →  4 adaptadores em models/
              └── Etapa 3 — Eval  →  resultados_avaliacao.csv + radar_comparativo.png
                    └── Etapa 4 — API   →  FastAPI em http://localhost:8000
```

### Modelos treinados

| Chave (`--model`) | Modelo base              | Tipo     | Pasta de saída               |
|-------------------|--------------------------|----------|------------------------------|
| `causal_1`        | EleutherAI/gpt-neo-125m  | Causal   | `models/lora_causal_model_1` |
| `causal_2`        | facebook/opt-1.3b        | Causal   | `models/lora_causal_model_2` |
| `seq2seq_1`       | google/mt5-small         | Seq2Seq  | `models/lora_seq2seq_model_1`|
| `seq2seq_2`       | facebook/bart-large      | Seq2Seq  | `models/lora_seq2seq_model_2`|

**Modelo gerador do dataset (Etapa 1):** `microsoft/phi-3-mini-4k-instruct` — usado apenas para gerar os pares instrução/resposta a partir dos chunks do PDF; não é fine-tunado.

---

## Estrutura do projeto

```
rag_lora_project/
├── main.py                            # servidor FastAPI (ponto de entrada da API)
├── compare.py                         # comparação qualitativa base vs. fine-tunado
├── makefile
├── requirements.txt
├── data/
│   ├── raw/                           # coloque o PDF-fonte aqui
│   ├── processed/                     # dataset_gerado.jsonl (gerado no stage1)
│   └── evaluation/                    # resultados_avaliacao.csv (gerado no stage3)
├── models/
│   ├── lora_causal_model_1/           # GPT-Neo 125M fine-tunado
│   ├── lora_causal_model_2/           # OPT-1.3B fine-tunado
│   ├── lora_seq2seq_model_1/          # mT5-Small fine-tunado
│   └── lora_seq2seq_model_2/          # BART-Large fine-tunado
├── reports/
│   └── radar_comparativo.png          # gerado no stage3
├── static/
│   └── index.html                     # frontend do chat
├── notebooks/                         # versões notebook (para entrega acadêmica)
│   ├── 01_rag.ipynb
│   ├── 02_lora.ipynb
│   └── 03_avaliacao_modelo_finetuned.ipynb
└── src/
    ├── api/
    │   ├── model_loader.py
    │   └── schemas.py
    ├── dataset_generation/
    │   ├── chunking.py
    │   ├── curate_dataset.py
    │   ├── extract_pdf.py
    │   ├── generate_qa.py
    │   └── run_stage1.py              # entry point CLI da etapa 1
    ├── evaluation/
    │   ├── bleu.py
    │   ├── evaluate_models.py         # entry point CLI da etapa 3
    │   ├── faithfulness.py
    │   ├── metrics.py
    │   ├── plan_adherence.py
    │   ├── relevance.py
    │   └── rouge.py
    ├── training/
    │   ├── lora_config.py
    │   ├── run_stage2.py              # entry point CLI da etapa 2
    │   └── trainer.py
    └── utils/
        ├── config.py                  # caminhos, hiperparâmetros LoRA, modelo gerador
        └── helpers.py
```

---

## Pré-requisitos

- **Python 3.10 ou superior** — [python.org](https://www.python.org/downloads/)
- **Make**
  - Windows: incluso no [Git for Windows](https://git-scm.com/download/win) (Git Bash) ou via `choco install make`
  - Linux/macOS: disponível por padrão

Confirme antes de começar:

```bash
python --version   # ou python3 --version
make --version
```

> Todos os comandos `make` devem ser executados **na raiz do projeto** — mesma pasta do `makefile`.

---

## Fluxo completo (do zero)

```
1. make setup      ← cria o ambiente virtual (.venv)
2. make install    ← instala as dependências do requirements.txt
3. make dirs       ← cria as pastas necessárias
4. make stage1     ← gera o dataset a partir do PDF
5. make stage2     ← treina os 4 modelos com LoRA
6. make stage3     ← avalia os modelos e gera CSV + gráfico radar
7. make api        ← sobe a API em http://localhost:8000
```

Cada etapa depende da anterior. Não pule nenhuma.

---

## Referência de comandos

### `make setup`

Cria o ambiente virtual Python (`.venv`) na raiz do projeto. Execute **uma única vez** por máquina.

```bash
make setup
```

Não é necessário ativar o ambiente manualmente — o makefile usa o Python da `.venv` em todos os outros comandos.

---

### `make install`

Instala as bibliotecas do `requirements.txt` dentro da `.venv` e baixa o tokenizador NLTK necessário para as métricas de avaliação.

```bash
make install
```

---

### `make dirs`

Cria toda a estrutura de pastas do projeto. Ignora as que já existem.

```bash
make dirs
```

> Coloque o PDF-fonte em `data/raw/` antes de rodar `make stage1`.

Pastas criadas:

```
data/raw/                      ← PDF-fonte
data/processed/                ← dataset_gerado.jsonl
data/evaluation/               ← CSV de resultados
models/lora_causal_model_1/    ← GPT-Neo 125M
models/lora_causal_model_2/    ← OPT-1.3B
models/lora_seq2seq_model_1/   ← mT5-Small
models/lora_seq2seq_model_2/   ← BART-Large
reports/                       ← gráfico radar
static/                        ← frontend HTML
```

---

### `make stage1`

Executa o script `src/dataset_generation/run_stage1.py`. Extrai o texto do PDF, divide em chunks (padrão: 500 caracteres, overlap de 50), gera pares instrução/resposta com o `phi-3-mini-4k-instruct` e salva o dataset.

```bash
make stage1
```

Antes de rodar: o PDF deve estar em `data/raw/`.

Resultado: `data/processed/dataset_gerado.jsonl` com ≥ 100 pares no formato `{"Instruction": "...", "Output": "..."}`.

Tempo estimado: 10–40 minutos (depende do PDF e da GPU disponível).

**Opções CLI disponíveis** (chamando o script diretamente):

```bash
# PDF específico
.venv/Scripts/python.exe -m src.dataset_generation.run_stage1 --pdf data/raw/arquivo.pdf

# Comparar dois tamanhos de chunk (recomendado — atende ao requisito da rubrica)
.venv/Scripts/python.exe -m src.dataset_generation.run_stage1 --chunk-size 300 --chunk-size 500
```

---

### `make stage2`

Executa o script `src/training/run_stage2.py`, treinando os 4 modelos com LoRA e salvando os adaptadores em `models/`.

```bash
make stage2
```

Antes de rodar: `make stage1` deve ter sido concluído.

Tempo estimado: várias horas em CPU. Recomendado rodar no Google Colab com GPU e copiar as pastas `models/` depois.

**Treinar apenas um modelo:**

```bash
.venv/Scripts/python.exe -m src.training.run_stage2 --model causal_1
# Opções: causal_1  causal_2  seq2seq_1  seq2seq_2
```

**Hiperparâmetros LoRA** (definidos em `src/utils/config.py`):

| Chave       | Modelo base              | r  | alpha | dropout | lr   | epochs | batch |
|-------------|--------------------------|----|----|---------|------|--------|-------|
| `causal_1`  | EleutherAI/gpt-neo-125m  | 16 | 32 | 0.10    | 2e-4 | 3      | 4     |
| `causal_2`  | facebook/opt-1.3b        | 8  | 16 | 0.05    | 1e-4 | 3      | 2     |
| `seq2seq_1` | google/mt5-small         | 16 | 32 | 0.10    | 3e-4 | 5      | 8     |
| `seq2seq_2` | facebook/bart-large      | 16 | 32 | 0.10    | 2e-4 | 3      | 4     |

---

### `make stage3`

Executa `src/evaluation/evaluate_models.py`, avaliando os 4 modelos com as métricas PPL, BLEU, ROUGE, Faithfulness, Answer Relevance e Plan Adherence.

```bash
make stage3
```

Antes de rodar: `make stage2` deve ter sido concluído.

Resultados:
- `data/evaluation/resultados_avaliacao.csv`
- `reports/radar_comparativo.png`

---

### `make stage3-nb`

Mesma avaliação do `stage3`, mas executando o notebook `03_avaliacao_modelo_finetuned.ipynb` com todas as saídas salvas inline — útil para entrega com gráficos renderizados.

```bash
make stage3-nb
```

---

### `make api`

Sobe o servidor FastAPI em primeiro plano na porta 8000. Modelos com pasta inexistente são pulados automaticamente — a API funciona mesmo com modelos parcialmente treinados.

```bash
make api
```

Após rodar, acesse:

| Endereço | Descrição |
|---|---|
| `http://localhost:8000` | Frontend de chat |
| `http://localhost:8000/docs` | Documentação Swagger |
| `http://localhost:8000/health` | Status dos modelos carregados |
| `http://localhost:8000/modelos` | Lista os modelos disponíveis (JSON) |

Para encerrar: `Ctrl + C`. O servidor usa `--reload`, então alterações em arquivos `.py` reiniciam automaticamente.

---

### `make api-bg` *(Linux/macOS apenas)*

Sobe a API em background, liberando o terminal. O PID é salvo em `.api.pid`.

```bash
make api-bg
# Para encerrar:
make stop-api
```

No Windows, use `make api` em um terminal separado.

---

### `make status`

Mostra o que já foi gerado — útil para conferir o progresso antes da entrega.

```bash
make status
```

Exemplo de saída:

```
-- Status do projeto --
[OK]    Dataset: data/processed/dataset_gerado.jsonl
[FALTA] Modelo causal 1 - rode: make stage2
[FALTA] Modelo causal 2 - rode: make stage2
[FALTA] Modelo seq2seq 1 - rode: make stage2
[FALTA] Modelo seq2seq 2 - rode: make stage2
[FALTA] Avaliacao - rode: make stage3
[FALTA] Grafico - rode: make stage3
```

---

### `make all`

Executa `install → dirs → stage1 → stage2 → stage3` em sequência.

```bash
make setup   # apenas se a .venv ainda não existe
make all
make api     # sobe a API em seguida
```

> `make setup` **não** está incluído no `all`. Rode-o antes se necessário.

---

### `make clean`

Remove `__pycache__`, `.pytest_cache` e `api.log`. Não apaga modelos nem dataset.

```bash
make clean
```

---

### `make clean-models`

Apaga as 4 pastas de adaptadores em `models/`. Use quando quiser re-treinar com hiperparâmetros diferentes sem regenerar o dataset.

```bash
make clean-models
make stage2   # re-treinar
```

---

### `make clean-data`

Apaga `data/processed/` com o `dataset_gerado.jsonl`. Use quando quiser testar um tamanho de chunk diferente.

```bash
make clean-data
make stage1   # regenerar o dataset
```

---

## Comparação qualitativa (base vs. fine-tunado)

O script `compare.py` gera respostas dos modelos base e fine-tunados lado a lado para perguntas de referência do glossário ANAC:

```bash
.venv/Scripts/python.exe compare.py
```

---

## Erros comuns

| Erro | Causa | Solução |
|---|---|---|
| `make: command not found` | Make não instalado | Instale via Git for Windows ou Chocolatey |
| `process_begin: CreateProcess failed` | `.venv` não existe | Rode `make setup` antes |
| `No such file or directory: requirements.txt` | Comando rodado fora da raiz | `cd` para a pasta do `makefile` |
| `ModuleNotFoundError: No module named 'src'` | Diretório errado ou `make install` não rodado | Confirme o diretório e rode `make install` |
| `Notebook timeout` | Notebook demorou mais que o limite | Aumente `--ExecutePreprocessor.timeout` no makefile (valor em segundos; 3600 = 1h) |

---

## Dependências principais

Listadas em `requirements.txt`. Destaques:

- `transformers >= 4.38.0` — modelos base e pipelines
- `peft >= 0.9.0` — adaptadores LoRA
- `datasets >= 2.18.0` — carregamento e tokenização do dataset
- `fastapi >= 0.110.0` + `uvicorn` — servidor da API
- `pdfplumber >= 0.10.0` — extração de texto do PDF
- `sacrebleu`, `rouge-score`, `nltk` — métricas de avaliação
- `matplotlib`, `seaborn` — gráfico radar
