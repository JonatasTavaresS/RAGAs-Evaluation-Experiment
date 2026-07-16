import os
import sys
import warnings
import json
import argparse
from unittest.mock import MagicMock
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def main():
    load_dotenv()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataset = os.path.join(BASE_DIR, "data", "ragas_dataset.json")
    default_output = os.path.join(BASE_DIR, "results", "eval_results.csv")

    parser = argparse.ArgumentParser(description="Avaliação do RAG usando RAGAS (Sem Ground Truth / Resposta de Referência)")
    parser.add_argument("--limit", type=int, default=None, help="Limite de amostras a avaliar para testes rápidos (ex: --limit 3)")
    parser.add_argument("--dataset", type=str, default=default_dataset, help="Caminho para o arquivo dataset de entrada JSON")
    parser.add_argument("--output", type=str, default=default_output, help="Arquivo CSV de saída dos resultados detalhados")
    args = parser.parse_args()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project or not location:
        print("Erro: GOOGLE_CLOUD_PROJECT e GOOGLE_CLOUD_LOCATION devem estar definidos no ambiente ou no arquivo .env")
        sys.exit(1)

    llm_vertex = ChatVertexAI(
        model_name="gemini-2.5-flash",
        temperature=0, 
        project=project,
        location=location,
        max_retries=3
    )

    embeddings_vertex = VertexAIEmbeddings(
        model_name="gemini-embedding-001",
        project=project,
        location=location
    )

    ragas_llm = LangchainLLMWrapper(llm_vertex)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings_vertex)

    dataset_file = args.dataset
    if not os.path.exists(dataset_file):
        print(f"Erro: O arquivo '{dataset_file}' não foi encontrado.")
        sys.exit(1)

    with open(dataset_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if args.limit:
        raw_data = raw_data[:args.limit]

    df_data = pd.DataFrame(raw_data)
    dataset_ragas = Dataset.from_pandas(df_data)

    print(f"Iniciando avaliação automática com RAGAS em {len(raw_data)} amostras...")
    print("Métricas utilizadas (Apenas Pergunta, Contexto e Resposta):")
    print(" - Faithfulness (Fidelidade ao Contexto)")
    print(" - AnswerRelevancy (Relevância da Resposta)")
    print(" - ContextRelevance (Relevância do Contexto)")
    print("-" * 50)

    # Executando a avaliação
    resultado = evaluate(
        dataset=dataset_ragas,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextRelevance()
        ],
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    df_res = resultado.to_pandas()

    # Salva resultados detalhados
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_res.to_csv(args.output, index=False)
    print(f"\nResultados detalhados salvos em '{args.output}'.")

    print("\n" + "=" * 60)
    print(" RESUMO DAS MÉTRICAS DE AVALIAÇÃO RAGAS (MÉDIA)")
    print("=" * 60)
    
    for col in ['faithfulness', 'answer_relevancy', 'context_relevance', 'nv_context_relevance']:
        if col in df_res.columns:
            mean_val = df_res[col].mean()
            display_name = col.replace('nv_', '').replace('_', ' ').title()
            print(f" * {display_name:<20}: {mean_val:.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print(" DETALHAMENTO DAS AMOSTRAS AVALIADAS")
    print("=" * 60)
    
    for idx, row in df_res.iterrows():
        print(f"\nAmostra #{idx + 1}")
        print(f"  Pergunta  : {row.get('user_input', row.get('question', ''))}")
        
        contexts = row.get('retrieved_contexts', row.get('contexts', []))
        print("  Contextos :")
        for c_idx, ctx in enumerate(contexts, 1):
            ctx_snippet = ctx[:150] + "..." if len(ctx) > 150 else ctx
            ctx_clean = ctx_snippet.replace('\n', ' ')
            print(f"    [{c_idx}] {ctx_clean}")
            
        print(f"  Resposta  : {row.get('response', row.get('answer', ''))}")
        print("  Notas     :")
        print(f"    - Faithfulness       : {row.get('faithfulness', 0.0):.4f}")
        print(f"    - Answer Relevancy   : {row.get('answer_relevancy', 0.0):.4f}")
        
        ctx_rel_score = row.get('context_relevance', row.get('nv_context_relevance', 0.0))
        print(f"    - Context Relevance  : {ctx_rel_score:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
