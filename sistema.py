"""Sistema de consulta para o modelo treinado."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from interface_app import render_console
from otimizacao import PARAMS_PADRAO, carregar_melhores_params
from prep import carregar, preparar_dataframe, criar_amostras, janelar, features_janela, SINAIS


class SistemaConsulta:
    def __init__(self, params=None, usar_otimizados=False) -> None:
        # Os parâmetros do Optuna são opt-in: a validação aninhada mostrou que eles
        # generalizam pior que o padrão neste conjunto (0.408 contra 0.438).
        if params is None:
            params = carregar_melhores_params() if usar_otimizados else dict(PARAMS_PADRAO)

        self.params = params
        self.modelo = RandomForestClassifier(**self.params)
        self.classes_ = None
        self._treinar()

    def _treinar(self) -> None:
        df = carregar()
        amostras = criar_amostras(df, modo="segmento")
        X = np.vstack(amostras["features"].to_list())
        y = amostras["classe"].to_numpy()
        self.modelo.fit(X, y)
        self.classes_ = self.modelo.classes_

    def consultar(self, trecho: pd.DataFrame) -> pd.DataFrame:
        trecho_prep = preparar_dataframe(trecho)
        if trecho_prep.empty:
            raise ValueError("O trecho não possui dados suficientes")

        amostras = criar_amostras(trecho_prep, modo="segmento")
        if amostras.empty:
            raise ValueError("O trecho não gerou amostras válidas")

        X = np.vstack(amostras["features"].to_list())
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
