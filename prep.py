"""Preprocessamento inicial dos dados."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
BANNER_CSV = DATA_DIR / "banner.csv"

# Tamanho da janela usada pelo modelo, em número de amostras consecutivas.
# O passo é metade da janela, o que gera 50% de sobreposição.
JANELA_TAMANHO = 180
JANELA_PASSO = JANELA_TAMANHO // 2

SINAIS = [
    "z_rms_velocity_mm_s",
    "temperature_c",
    "x_rms_velocity_mm_s",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
    "rpm",
]


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


def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "fault" in df.columns:
        dt = df["created_at"].diff().dt.total_seconds()
        raw_label_changed = df["fault"].ne(df["fault"].shift())
        new_segment = raw_label_changed | dt.gt(3600)
        df["segment_id"] = new_segment.cumsum()
    else:
        df["segment_id"] = 0

    if "fault" in df.columns:
        df["classe"] = df["fault"].apply(classe_base)
        df = df.dropna(subset=["classe"]).reset_index(drop=True)
    else:
        df["classe"] = pd.NA

    feature_cols = []
    if "created_at" in df.columns:
        feature_cols.append("created_at")
    feature_cols.extend([*SINAIS, "classe", "segment_id"])
    df = df.loc[:, feature_cols]

    return df


def carregar():
    df = pd.read_csv(BANNER_CSV, parse_dates=["created_at"])
    return preparar_dataframe(df)


def carregar_dados():
    return carregar()


def _features_resumo(dados: pd.DataFrame) -> np.ndarray:
    valores = dados[SINAIS].to_numpy(dtype=float)
    n = len(dados)

    if n <= 1:
        t = np.array([0.0] * n, dtype=float)
        denom = 1.0
    else:
        t = np.arange(n) - (n - 1) / 2
        denom = float(np.dot(t, t))

    features = []
    for coluna in range(valores.shape[1]):
        x = valores[:, coluna]
        slope = float(np.dot(t, x) / denom) if denom > 0 else 0.0
        features.extend(
            [
                float(np.median(x)),
                float(np.std(x)),
                slope,
                float(np.max(x) - np.min(x)),
                float(np.percentile(x, 90) - np.percentile(x, 10)),
            ]
        )

    return np.array(features, dtype=float)


def features_janela(janela: pd.DataFrame, tamanho=JANELA_TAMANHO) -> np.ndarray:
    if len(janela) != tamanho:
        raise ValueError(f"A janela deve conter exatamente {tamanho} amostras")
    return _features_resumo(janela)


def features_segmento(segmento: pd.DataFrame) -> np.ndarray:
    if segmento.empty:
        raise ValueError("O segmento deve conter pelo menos uma amostra")
    return _features_resumo(segmento)


def criar_amostras(df: pd.DataFrame, modo="segmento", tamanho=JANELA_TAMANHO, passo=JANELA_PASSO):
    modo = str(modo).lower()
    amostras = []

    for segment_id, grupo in df.groupby("segment_id", sort=True):
        if grupo.empty:
            continue

        if modo in {"segmento", "sem_janela", "segmento_completo"}:
            amostras.append(
                {
                    "features": features_segmento(grupo),
                    "classe": grupo["classe"].iloc[0],
                    "segment_id": int(segment_id),
                    "dados_brutos": grupo[SINAIS].copy(),
                }
            )
            continue

        if len(grupo) < tamanho:
            continue

        for inicio in range(0, len(grupo) - tamanho + 1, passo):
            janela = grupo.iloc[inicio : inicio + tamanho]
            amostras.append(
                {
                    "features": features_janela(janela, tamanho=tamanho),
                    "classe": janela["classe"].iloc[0],
                    "segment_id": int(segment_id),
                    "janela_bruta": janela[SINAIS].copy(),
                }
            )

    return pd.DataFrame(amostras)


def janelar(df: pd.DataFrame, tamanho=JANELA_TAMANHO, passo=JANELA_PASSO):
    return criar_amostras(df, modo="janela", tamanho=tamanho, passo=passo)


if __name__ == "__main__":
    df = carregar()
    print(f"Arquivo pronto em: {BANNER_CSV}")
    print(df.shape)
    print(df.isna().sum().sum())
