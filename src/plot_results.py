import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(BASE_DIR, "results", "eval_results.csv")
    default_output_dir = os.path.join(BASE_DIR, "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Erro: O arquivo de entrada '{args.input}' não existe.")
        sys.exit(1)

    df = pd.read_csv(args.input)

    cols = []
    labels = []
    if 'faithfulness' in df.columns:
        cols.append('faithfulness')
        labels.append('Fidelidade')
    if 'answer_relevancy' in df.columns:
        cols.append('answer_relevancy')
        labels.append('Relevância\nda Resposta')
    if 'nv_context_relevance' in df.columns:
        cols.append('nv_context_relevance')
        labels.append('Relevância\ndo Contexto')
    elif 'context_relevance' in df.columns:
        cols.append('context_relevance')
        labels.append('Relevância\ndo Contexto')

    if not cols:
        print("Erro: Nenhuma métrica encontrada no CSV.")
        sys.exit(1)

    df_metrics = df[cols]

    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    ax = sns.boxplot(data=df_metrics, palette="Set1", width=0.5)
    plt.ylabel('Pontuação', color='black')
    plt.xticks(ticks=range(len(cols)), labels=labels, color='black')
    plt.yticks(color='black')
    plt.ylim(-0.1, 1.1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    sns.despine()
    plt.savefig(os.path.join(args.output_dir, 'boxplot_metricas.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    corr_matrix = df_metrics.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='Blues',
                vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.xticks(ticks=[x + 0.5 for x in range(len(cols))],
               labels=labels, rotation=0)
    plt.yticks(ticks=[x + 0.5 for x in range(len(cols))],
               labels=labels, rotation=0)
    plt.savefig(os.path.join(args.output_dir, 'matriz_correlacao.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gráficos gerados com sucesso em '{args.output_dir}'!")


if __name__ == "__main__":
    main()
