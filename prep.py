"""Preprocessamento inicial dos dados."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dados"
BANNER_CSV = DATA_DIR / "banner.csv"


def carregar_dados():
    return pd.read_csv(BANNER_CSV)


if __name__ == "__main__":
    print(f"Arquivo pronto em: {BANNER_CSV}")
