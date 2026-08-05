"""Interface simples para visualizar resultados do projeto."""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from prep import carregar


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    top_faults = df["fault"].value_counts().head(5).to_dict() if "fault" in df.columns else {}
    return {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "valores_ausentes": int(df.isna().sum().sum()),
        "ordenado": bool(df["created_at"].is_monotonic_increasing),
        "top_faults": top_faults,
        "primeiras_linhas": df.head(3).to_string(index=False),
    }


def render_console() -> None:
    while True:
        print("\n=== MENU DE VISUALIZAÇÃO ===")
        print("1 - Resumo geral")
        print("2 - Top faults")
        print("3 - Abrir versão web")
        print("0 - Sair")

        choice = input("Escolha uma opção: ").strip()
        if choice == "1":
            df = carregar()
            summary = build_summary(df)
            print("\n" + "=" * 60)
            print("RESUMO GERAL")
            print("=" * 60)
            print(f"Linhas: {summary['linhas']}")
            print(f"Colunas: {summary['colunas']}")
            print(f"Valores ausentes: {summary['valores_ausentes']}")
            print(f"Dados ordenados por created_at: {'sim' if summary['ordenado'] else 'não'}")
            print("\nPreview das primeiras linhas:")
            print(summary["primeiras_linhas"])
            print("=" * 60)
        elif choice == "2":
            df = carregar()
            print("\nTop faults:")
            for fault, count in df["fault"].value_counts().head(10).items():
                print(f"  - {fault}: {count}")
        elif choice == "3":
            print("Abra o arquivo app.py com Streamlit para ver a interface web.")
            print("Comando: streamlit run app.py")
        elif choice == "0":
            print("Encerrando...")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    render_console()
