"""Preprocessamento inicial dos dados."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
BANNER_CSV = DATA_DIR / "banner.csv"


def carregar():
    df = pd.read_csv(BANNER_CSV, parse_dates=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def carregar_dados():
    return carregar()


if __name__ == "__main__":
    df = carregar()
    print(f"Arquivo pronto em: {BANNER_CSV}")
    print(df.shape)
    print(df.isna().sum().sum())
