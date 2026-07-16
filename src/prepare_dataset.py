import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "relatorio_consumo.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "ragas_dataset.json")

def process_context(contexto_enviado):
    if not contexto_enviado:
        return ["Sem contexto de ferramentas."]
    
    tool_results = contexto_enviado.get("tool_results", [])
    if not tool_results:
        return ["Nenhuma ferramenta executada."]
    
    contexts_list = []
    for idx, tool in enumerate(tool_results, 1):
        func = tool.get("function", "")
        args = tool.get("args", {})
        args_clean = {k: v for k, v in args.items() if k not in ("token", "building_id")}
        result = tool.get("result", {})
        
        ctx_str = f"Chamada {idx}: {func}({args_clean}) -> {result}"
        contexts_list.append(ctx_str)
        
    return contexts_list

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo de entrada '{INPUT_FILE}' não encontrado.")
        return

    print(f"Lendo '{INPUT_FILE}'...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ragas_data = []
    for item in data:
        question = item.get("pergunta", "")
        answer = item.get("resposta", "")
        contexto_enviado = item.get("contexto_enviado", {})
        
        contexts = process_context(contexto_enviado)
        
        ragas_data.append({
            "question": question,
            "contexts": contexts,
            "answer": answer
        })
    
    print(f"Gerando dataset em '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ragas_data, f, ensure_ascii=False, indent=2)
        
    print(f"Sucesso! {len(ragas_data)} registros processados e salvos.")

if __name__ == "__main__":
    main()
