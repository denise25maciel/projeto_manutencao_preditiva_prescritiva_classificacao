"""Preprocessamento inicial dos dados."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
BANNER_CSV = DATA_DIR / "banner.csv"


def classe_base(rótulo: str) -> str:
    if pd.isna(rótulo):
        return pd.NA

    label = str(rótulo)
    label = label.strip()

    discarded = {"teste", "acelerando", "new_tes", "new_teste"}
    if label in discarded:
        return pd.NA

    replacements = {
        "desabalanceado": "desbalanceado",
        "desbanlanceado": "desbalanceado",
        "ddesbalanceado": "desbalanceado",
        "dedesbalanceado": "desbalanceado",
        "desabanceado": "desbalanceado",
        "desbalanceamento": "desbalanceado",
        "normla": "normal",
        "mortor_desligado": "motor_desligado",
        "cockecocked": "cocked",
        "rolamento_comb": "rolamento_combination",
    }

    for source, target in replacements.items():
        if source in label:
            label = label.replace(source, target)

    label = label.removeprefix("new_")

    if label.startswith("baseline"):
        return "normal"

    tipos = [
        "motor_desligado",
        "falta_fase",
        "rolamento_inner",
        "rolamento_outer",
        "rolamento_ball",
        "rolamento_combination",
        "eccentric",
        "cocked",
        "desbalanceado",
        "desalinhado",
        "normal",
        "baseline",
        "polia",
        "correia",
        "ventoinha",
    ]

    for tipo in tipos:
        if label.startswith(tipo):
            return tipo

    return label


def carregar():
    df = pd.read_csv(BANNER_CSV, parse_dates=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    if "fault" in df.columns:
        df = df.copy()
        dt = df["created_at"].diff().dt.total_seconds()
        raw_label_changed = df["fault"].ne(df["fault"].shift())
        new_segment = raw_label_changed | dt.gt(3600)
        df["segment_id"] = new_segment.cumsum()
    else:
        df["segment_id"] = 0

    df["classe"] = df["fault"].apply(classe_base)
    df = df.dropna(subset=["classe"]).reset_index(drop=True)

    return df


def carregar_dados():
    return carregar()


if __name__ == "__main__":
    df = carregar()
    print(f"Arquivo pronto em: {BANNER_CSV}")
    print(df.shape)
    print(df.isna().sum().sum())
