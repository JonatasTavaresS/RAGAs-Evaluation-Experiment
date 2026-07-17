# Avaliação Automatizada de LLMs no Monitoramento de Utilidades: Uma Aplicação do Framework RAGAs

Este projeto realiza a avaliação de um sistema de *Retrieval Augmented Generation* (RAG) utilizando a biblioteca **RAGAs** (*Retrieval Augmented Generation Assessment*) e modelos de LLM e Embeddings hospedados no Google Cloud Vertex AI (Gemini).

A estrutura do projeto foi organizada segundo padrões e boas práticas de reprodutibilidade para permitir que qualquer pessoa execute os testes e recrie os resultados obtidos sem dificuldades.

Este projeto é uma reaplicação do artigo publicado em [ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/).

A atividade final desenvolvida para a disciplina está disponível [neste link do Google Drive](https://drive.google.com/file/d/1377f_exB9N9WBQB2o0sNraGeHRxvOBJ2/view?usp=drive_link).

---

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

```text
RAGAs-Evaluation-Experiment/
 ├── data/                 # Dados de entrada e datasets gerados
 │   ├── relatorio_consumo.json                      # Relatório bruto (dados de entrada)
 │   └── ragas_dataset.json                          # Dataset processado gerado para o RAGAs
 ├── src/                  # Código-fonte Python
 │   ├── __init__.py       # Define 'src' como um pacote Python
 │   ├── prepare_dataset.py # Script que processa o arquivo bruto em um dataset RAGAs
 │   ├── eval.py           # Script principal de avaliação usando Ragas e Vertex AI
 │   └── plot_results.py   # Script que gera o boxplot e a matriz de correlação das métricas
 ├── scripts/              # Scripts shell utilitários para execução rápida
 │   ├── run_prepare.sh    # Executa a preparação dos dados
 │   ├── run_eval.sh       # Executa a avaliação RAGAs
 │   └── run_plot.sh       # Executa a geração dos gráficos
 ├── results/              # Resultados gerados pela execução
 │   ├── eval_results.csv  # Resultados detalhados da avaliação
 │   ├── boxplot_metricas.png # Boxplot das métricas avaliadas
 │   └── matriz_correlacao.png # Matriz de correlação das métricas
 ├── pyproject.toml        # Metadados do projeto e declaração de dependências
 ├── requirements.txt      # Arquivo de dependências compilado para uso com pip
 ├── uv.lock               # Lockfile do uv para reprodutibilidade estrita
 └── README.md             # Este arquivo de instruções
```

---

## Pré-requisitos

Para executar este projeto, você precisará de:

1. **Python 3.13** ou superior.
2. Uma conta no **Google Cloud Platform (GCP)** com acesso à **Vertex AI API** ativada.
3. Credenciais configuradas no seu ambiente de execução.

### Autenticação no Google Cloud
Antes de executar a avaliação, garanta que suas credenciais estão configuradas. Você pode fazer a autenticação usando a CLI do Google Cloud:

```bash
gcloud auth application-default login
```

Para mais instruções de configuração e autenticação no GCP, consulte a [documentação oficial do Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/gcp-auth?hl=pt-br). Para testar e experimentar com modelos multimodais de forma visual, acesse o [Agent Studio Multimodal no Console do GCP](https://console.cloud.google.com/agent-platform/studio/multimodal).

Além disso, configure as variáveis de ambiente com o ID do seu projeto do GCP e a região onde os modelos do Vertex AI estão disponíveis. Você pode configurar essas variáveis no seu terminal ou criar um arquivo `.env` a partir do template `.env.example`:

```bash
export GOOGLE_CLOUD_PROJECT="seu-projeto-gcp"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

---

## Instalação e Configuração

Você pode configurar o ambiente utilizando o gerenciador rápido de pacotes `uv` (recomendado) ou por meio do `pip` tradicional com ambientes virtuais nativos.

### Instalação do `uv` no Ubuntu
Para instalar o `uv` pela primeira vez no Ubuntu, execute o comando de instalação oficial:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Após o término da instalação, atualize o seu ambiente executando:

```bash
source $HOME/.local/bin/env
```

Para certificar-se de que a instalação foi concluída com sucesso:

```bash
uv --version
```

### Opção A: Usando `uv` (Recomendado)
Ao utilizar o `uv`, não há necessidade de criar ou ativar ambientes virtuais manualmente para executar os scripts. O comando `uv run` gerencia automaticamente a criação do ambiente virtual e a sincronização das dependências definidas no projeto.

### Opção B: Usando `pip` e Ambiente Virtual Nativo
Se preferir a instalação tradicional do Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Como Executar os Experimentos

Os passos abaixo detalham o fluxo completo para reproduzir os experimentos. Escolha os comandos de acordo com o gerenciador de pacotes escolhido.

### Passo 1: Preparação do Dataset
O primeiro script lê o relatório bruto com as interações (`data/relatorio_consumo.json`), extrai a pergunta, a resposta formulada e o contexto de ferramentas utilizado, salvando os dados no formato esperado pelo RAGAs em `data/ragas_dataset.json`.

* **Com `uv`**:
  ```bash
  uv run src/prepare_dataset.py
  ```
* **Com ambiente virtual tradicional (ativado)**:
  ```bash
  python src/prepare_dataset.py
  ```
* **Via script utilitário**:
  ```bash
  ./scripts/run_prepare.sh
  ```

### Passo 2: Execução da Avaliação (RAGAs)
Com o dataset gerado, a avaliação roda as métricas do RAGAs utilizando o Vertex AI (Gemini).

* **Com `uv`**:
  ```bash
  uv run src/eval.py
  ```
* **Com ambiente virtual tradicional (ativado)**:
  ```bash
  python src/eval.py
  ```
* **Via script utilitário**:
  ```bash
  ./scripts/run_eval.sh
  ```

#### Testes Rápidos (Limite de Amostras)
A avaliação completa pode demorar alguns minutos. Para fazer um teste rápido e garantir que a integração com o Vertex AI está funcionando, você pode limitar o número de amostras usando o argumento `--limit`:

* **Com `uv`**:
  ```bash
  uv run src/eval.py --limit 2
  ```
* **Com ambiente virtual tradicional**:
  ```bash
  python src/eval.py --limit 2
  ```
* **Via script utilitário**:
  ```bash
  ./scripts/run_eval.sh --limit 2
  ```

---

### Passo 3: Geração de Gráficos
Com os resultados detalhados exportados em `results/eval_results.csv`, você pode gerar o boxplot e a matriz de correlação das métricas.

* **Com `uv`**:
  ```bash
  uv run src/plot_results.py
  ```
* **Com ambiente virtual tradicional (ativado)**:
  ```bash
  python src/plot_results.py
  ```
* **Via script utilitário**:
  ```bash
  ./scripts/run_plot.sh
  ```

Os gráficos gerados serão salvos na pasta `results/`:
* `results/boxplot_metricas.png`: Distribuição das notas de fidelidade, relevância de resposta e contexto.
* `results/matriz_correlacao.png`: Matriz de correlação de Spearman entre as métricas.

---

## Métricas Avaliadas e Resultados

O pipeline avalia as respostas do RAG utilizando 3 métricas fundamentais que não requerem *Ground Truth*:

1. **Faithfulness (Fidelidade)**: Avalia se a resposta gerada é baseada unicamente nas informações presentes no contexto recuperado.
2. **Answer Relevancy (Relevância da Resposta)**: Avalia o quão diretamente a resposta gerada responde à pergunta feita.
3. **Context Relevance (Relevância do Contexto)**: Avalia a proporção de informações úteis/relevantes presentes no contexto recuperado em relação ao total de informações contidas nele.

### Resultados Gerados
Ao finalizar a avaliação, o console mostrará uma tabela de médias de cada métrica e o detalhamento amostra a amostra.
Todos os dados detalhados (incluindo as notas de cada amostra) são exportados para:
* **`results/eval_results.csv`**
