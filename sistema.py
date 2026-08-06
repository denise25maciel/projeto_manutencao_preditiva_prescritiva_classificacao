"""Sistema de consulta para o modelo treinado."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from interface_app import render_console
from prep import carregar, preparar_dataframe, janelar, features_janela, SINAIS


class SistemaConsulta:
    def __init__(self) -> None:
        self.modelo = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
        self.classes_ = None
        self._treinar()

    def _treinar(self) -> None:
        df = carregar()
        janelas = janelar(df)
        X = np.vstack(janelas["features"].to_list())
        y = janelas["classe"].to_numpy()
        self.modelo.fit(X, y)
        self.classes_ = self.modelo.classes_

    def consultar(self, trecho: pd.DataFrame) -> pd.DataFrame:
        trecho_prep = preparar_dataframe(trecho)
        if trecho_prep.empty:
            raise ValueError("O trecho não possui dados suficientes")

        janelas = janelar(trecho_prep)
        if janelas.empty:
            raise ValueError("O trecho não gerou janelas válidas")

        X = np.vstack(janelas["features"].to_list())
        probs = self.modelo.predict_proba(X)
        media = probs.mean(axis=0)
        ranking = pd.DataFrame(
            {"classe": self.classes_, "probabilidade_media": media}
        ).sort_values("probabilidade_media", ascending=False)
        return ranking.reset_index(drop=True)


def main():
    render_console()


if __name__ == "__main__":
    main()
