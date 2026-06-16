# Tutorial do Makefile — RAG + LoRA Project

Este guia explica **o que é o Makefile**, **como ele funciona** e **como usar cada comando** do projeto. Leia do início ao fim na primeira vez; depois use como referência rápida.

---

## O que é um Makefile?

Um Makefile é um arquivo de automação. Em vez de digitar comandos longos no terminal toda vez, você define atalhos chamados **targets** (alvos). Você chama um target assim:

```cmd
make nome-do-target
```

O Make lê o `Makefile` na pasta atual e executa os comandos associados àquele target. É equivalente a ter um script, mas com suporte a dependências entre etapas.

---

## Pré-requisitos

Antes de usar qualquer comando, você precisa ter instalado:

- **Python 3.10 ou superior** — baixe em [python.org](https://www.python.org/downloads/)
- **Make para Windows** — instalado junto com o [Git for Windows](https://git-scm.com/download/win) (vem no Git Bash) ou via [Chocolatey](https://chocolatey.org/): `choco install make`

Para confirmar que ambos estão disponíveis, abra o terminal e rode:

```cmd
python --version
make --version
```

Os dois precisam retornar uma versão, sem erro.

> **Importante:** todos os comandos `make` devem ser rodados **dentro da pasta raiz do projeto** — a mesma onde o `Makefile` está.

---

## Fluxo completo do zero

Se você acabou de clonar o repositório, siga esta ordem obrigatória:

```
1. make setup      ← cria o ambiente virtual (.venv)
2. make install    ← instala todas as dependências dentro da .venv
3. make dirs       ← cria as pastas do projeto
4. make stage1     ← gera o dataset a partir do PDF
5. make stage2     ← treina os 4 modelos com LoRA
6. make stage3     ← avalia os modelos e gera tabela + gráfico
7. make api        ← sobe a API para testar no navegador
```

Cada etapa depende da anterior. Não pule nenhuma.

---

## Referência de todos os comandos

### `make setup`

**O que faz:** cria o ambiente virtual Python (`.venv`) na raiz do projeto.

```cmd
make setup
```

Precisa ser rodado **uma única vez** por máquina. Depois de rodar, você verá a pasta `.venv/` criada. Não é necessário ativar o ambiente manualmente — o Makefile já usa o Python da `.venv` diretamente em todos os outros comandos.

Quando rodar de novo: nunca, a menos que você delete a `.venv` ou mude de máquina.

---

### `make install`

**O que faz:** instala todas as bibliotecas listadas no `requirements.txt` dentro da `.venv`, e baixa o tokenizador do NLTK necessário para as métricas.

```cmd
make install
```

Precisa que o `make setup` já tenha sido executado (a `.venv` precisa existir) e que o arquivo `requirements.txt` esteja na raiz do projeto.

Quando rodar de novo: sempre que o `requirements.txt` for atualizado com novas dependências.

---

### `make dirs`

**O que faz:** cria toda a estrutura de pastas do projeto que ainda não existe.

```cmd
make dirs
```

Cria as seguintes pastas:

```
data/
  raw/           ← coloque o PDF-fonte aqui antes do stage1
  processed/     ← o dataset_gerado.jsonl será salvo aqui
  evaluation/    ← CSV de resultados da avaliação
models/
  lora_causal_model_1/    ← GPT-2 Medium treinado
  lora_causal_model_2/    ← OPT-1.3B treinado
  lora_seq2seq_model_1/   ← Flan-T5 treinado
  lora_seq2seq_model_2/   ← BART Large treinado
reports/         ← gráfico radar
static/          ← frontend HTML da API
```

Quando rodar de novo: não é necessário; o comando já ignora pastas que existem.

---

### `make stage1`

**O que faz:** executa o notebook `notebooks/01_rag.ipynb` automaticamente, gerando o dataset de pares instrução/resposta a partir do PDF.

```cmd
make stage1
```

**Antes de rodar:** coloque o PDF-fonte dentro de `data/raw/`. Sem o PDF, o notebook vai quebrar na célula de extração de texto.

**Resultado:** arquivo `data/processed/dataset_gerado.jsonl` com pelo menos 100 pares no formato:
```json
{"Instruction": "...", "Output": "..."}
```

O notebook é salvo com todas as saídas visíveis (células executadas), que é o que o professor exige para a entrega.

Tempo estimado: 10–40 minutos dependendo do tamanho do PDF e se há GPU disponível.

---

### `make stage2`

**O que faz:** executa o notebook `notebooks/02_lora.ipynb`, treinando os 4 modelos com LoRA.

```cmd
make stage2
```

**Antes de rodar:** o `make stage1` precisa ter sido concluído (o `dataset_gerado.jsonl` precisa existir).

**Resultado:** 4 pastas em `models/`, cada uma com os pesos do adaptador LoRA:

```
models/lora_causal_model_1/   → GPT-2 Medium fine-tunado
models/lora_causal_model_2/   → OPT-1.3B fine-tunado
models/lora_seq2seq_model_1/  → Flan-T5 fine-tunado
models/lora_seq2seq_model_2/  → BART Large fine-tunado
```

Tempo estimado: várias horas em CPU. Se possível, rode no Google Colab com GPU e copie as pastas `models/` para cá depois.

---

### `make stage3`

**O que faz:** avalia os 4 modelos treinados usando o script Python diretamente (sem abrir notebook), calculando PPL, BLEU, ROUGE, Faithfulness, Answer Relevance e Plan Adherence.

```cmd
make stage3
```

**Antes de rodar:** o `make stage2` precisa ter sido concluído (os modelos precisam existir).

**Resultado:**
- `data/evaluation/resultados_avaliacao.csv` — tabela com todas as métricas
- `reports/radar_comparativo.png` — gráfico radar dos 4 modelos

---

### `make stage3-nb`

**O que faz:** mesma avaliação do `stage3`, mas executando o notebook `03_avaliacao_modelo_finetuned.ipynb` com todas as saídas salvas inline — útil para incluir no relatório com os gráficos renderizados.

```cmd
make stage3-nb
```

Use este quando precisar entregar o notebook com saídas visíveis. Use o `stage3` quando quiser só os resultados rápido.

---

### `make api`

**O que faz:** sobe o servidor FastAPI em primeiro plano na porta 8000.

```cmd
make api
```

Após rodar, acesse no navegador:

- **Frontend de chat:** `http://localhost:8000`
- **Documentação Swagger:** `http://localhost:8000/docs`
- **Health check:** `http://localhost:8000/health`

Para encerrar: pressione `Ctrl + C` no terminal.

O servidor usa `--reload`, então qualquer alteração nos arquivos `.py` reinicia automaticamente — útil durante desenvolvimento.

---

### `make api-bg` *(Linux/macOS apenas)*

**O que faz:** sobe a API em background, liberando o terminal.

```cmd
make api-bg
```

O PID do processo é salvo em `.api.pid`. Para encerrar:

```cmd
make stop-api
```

No Windows, use `make api` em um terminal separado.

---

### `make status`

**O que faz:** mostra um resumo do que já foi gerado no projeto — útil para conferir o progresso antes da entrega.

```cmd
make status
```

Exemplo de saída:

```
-- Status do projeto --
OK  Dataset gerado
--  Modelo causal 1 NAO treinado
--  Modelo causal 2 NAO treinado
--  Modelo seq2seq 1 NAO treinado
--  Modelo seq2seq 2 NAO treinado
--  Avaliacao NAO realizada: rode make stage3
--  Grafico NAO gerado
```

---

### `make all`

**O que faz:** roda `install → dirs → stage1 → stage2 → stage3` em sequência.

```cmd
make all
```

Use quando quiser rodar o pipeline completo de uma vez. Não inclui o `make api` — você sobe a API separadamente depois.

> Atenção: o `make setup` **não** está incluído no `all`. Rode `make setup` antes do `make all` se a `.venv` ainda não existir.

---

### `make clean`

**O que faz:** remove arquivos temporários — pastas `__pycache__`, `.pytest_cache` e o log da API (`api.log`). Não apaga modelos nem dataset.

```cmd
make clean
```

---

### `make clean-models`

**O que faz:** apaga as 4 pastas de modelos treinados em `models/`. Use quando quiser re-treinar do zero com hiperparâmetros diferentes, sem precisar regenerar o dataset.

```cmd
make clean-models
```

Depois rode `make stage2` para treinar novamente.

---

### `make clean-data`

**O que faz:** apaga a pasta `data/processed/` com o `dataset_gerado.jsonl`. Use quando quiser testar um tamanho de chunk diferente no `01_rag.ipynb`.

```cmd
make clean-data
```

Depois rode `make stage1` para regenerar o dataset.

---

## Erros comuns e como resolver

**`make: command not found`**
O Make não está instalado. Instale via Git for Windows ou Chocolatey.

**`process_begin: CreateProcess failed`**
A `.venv` não existe. Rode `make setup` antes do `make install`.

**`No such file or directory: requirements.txt`**
O `requirements.txt` não está na raiz do projeto. Verifique com `dir` (Windows) se o arquivo aparece na mesma pasta do `Makefile`.

**`ModuleNotFoundError: No module named 'src'`**
O comando foi rodado fora da raiz do projeto, ou a `.venv` não está com as dependências instaladas. Confirme que está na pasta certa com `cd` e que rodou `make install`.

**`Notebook timeout`**
O notebook demorou mais que o limite configurado. Para aumentar o tempo, edite o Makefile e mude o valor de `--ExecutePreprocessor.timeout` na etapa correspondente. O valor está em segundos (3600 = 1 hora, 86400 = 24 horas).
