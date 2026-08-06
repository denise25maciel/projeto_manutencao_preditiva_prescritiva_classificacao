"""Avaliação do sistema com validação cruzada."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score

from prep import carregar, criar_amostras


def avaliar(n_estimators=400, modo="segmento"):
    """Valida o modelo com as duas estratégias de divisão."""
    df = carregar()
    amostras = criar_amostras(df, modo=modo)

    X = np.vstack(amostras["features"].to_list())
    y = amostras["classe"].to_numpy()
    groups = amostras["segment_id"].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    resultados = []
    for splitter, name in [(skf, "aleatoria"), (sgkf, "por_segmento")]:
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            score = accuracy_score(y[test_idx], preds)
            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "accuracy": float(score),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "y_true": y[test_idx],
                    "y_pred": preds,
                    "teste_amostras": amostras.iloc[test_idx].copy(),
                }
            )

        resultados.append(
            {
                "nome": name,
                "folds": fold_results,
                "media": float(np.mean([item["accuracy"] for item in fold_results])),
            }
        )

    print("Resultados da validação:")
    for item in resultados:
        print(f"{item['nome']}: {item['media']:.3f}")

    return resultados


if __name__ == "__main__":
    avaliar()
