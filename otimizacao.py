"""Busca de hiperparâmetros com Optuna.

A busca otimiza a acurácia da validação por segmento (StratifiedGroupKFold), não a
da validação aleatória. A aleatória vaza janelas do mesmo segmento entre treino e
teste, então otimizá-la escolheria os hiperparâmetros que melhor memorizam segmento.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from prep import carregar, criar_amostras

# O Optuna só é necessário para rodar a busca. app.py e sistema.py importam este
# módulo apenas para ler os parâmetros já salvos, e não devem quebrar sem ele.
try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


def _exigir_optuna():
    if optuna is None:
        raise ImportError(
            "A busca de hiperparâmetros exige o Optuna. Instale com: pip install optuna"
        )

PARAMS_JSON = Path(__file__).resolve().parent / "melhores_params.json"

N_SPLITS = 5
SEED = 0

PARAMS_PADRAO = {
    "n_estimators": 400,
    "random_state": SEED,
    "n_jobs": -1,
}


def montar_dados(modo="janela"):
    """Devolve X, y e os grupos (segment_id) prontos para a busca."""
    df = carregar()
    amostras = criar_amostras(df, modo=modo)

    X = np.vstack(amostras["features"].to_list())
    y = amostras["classe"].to_numpy()
    groups = amostras["segment_id"].to_numpy()

    return X, y, groups


def _max_features(valor):
    """Converte a escolha categórica em valor aceito pelo sklearn."""
    if valor in {"sqrt", "log2"}:
        return valor
    return float(valor)


def espaco_busca(trial: optuna.Trial) -> dict:
    """Espaço de busca para o RandomForest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": _max_features(
            trial.suggest_categorical("max_features", ["sqrt", "log2", "0.3", "0.5"])
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["none", "balanced", "balanced_subsample"]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    # profundidade ilimitada é uma opção legítima, não um valor extremo do intervalo
    if trial.suggest_categorical("limitar_profundidade", [True, False]):
        params["max_depth"] = trial.suggest_int("max_depth", 3, 40)
    else:
        params["max_depth"] = None

    if params["class_weight"] == "none":
        params["class_weight"] = None

    params["random_state"] = SEED
    params["n_jobs"] = -1

    return params


def _acuracia_cv(params, X, y, groups, n_splits=N_SPLITS, trial=None):
    """Acurácia média em validação por segmento. Reporta folds ao Optuna para poda."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    scores = []
    for passo, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        modelo = RandomForestClassifier(**params)
        modelo.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], modelo.predict(X[test_idx])))

        # poda: interrompe trials que já se mostraram ruins nos primeiros folds
        if trial is not None:
            trial.report(float(np.mean(scores)), passo)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(scores)), float(np.std(scores))


def objetivo(trial, X, y, groups):
    params = espaco_busca(trial)
    media, desvio = _acuracia_cv(params, X, y, groups, trial=trial)
    trial.set_user_attr("desvio_folds", desvio)
    return media


def otimizar(n_trials=60, modo="janela", salvar=True, mostrar=True):
    """Roda a busca e devolve (study, melhores_params)."""
    _exigir_optuna()
    X, y, groups = montar_dados(modo=modo)

    if mostrar:
        print(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features, "
              f"{len(np.unique(groups))} segmentos, {len(np.unique(y))} classes")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        study_name=f"rf_{modo}",
    )
    study.optimize(lambda t: objetivo(t, X, y, groups), n_trials=n_trials, show_progress_bar=False)

    melhores = espaco_busca(optuna.trial.FixedTrial(study.best_params))

    if mostrar:
        podados = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"\nTrials: {len(study.trials)} ({podados} podados)")
        print(f"Melhor acurácia (por segmento): {study.best_value:.4f}")
        print(f"Desvio entre folds: {study.best_trial.user_attrs.get('desvio_folds', float('nan')):.4f}")
        print("\nMelhores hiperparâmetros:")
        for chave, valor in sorted(study.best_params.items()):
            print(f"  {chave}: {valor}")

    if salvar:
        PARAMS_JSON.write_text(
            json.dumps({"modo": modo, "acuracia_busca": study.best_value,
                        "params": {k: v for k, v in melhores.items()}},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if mostrar:
            print(f"\nSalvo em: {PARAMS_JSON.name}")

    return study, melhores


def carregar_melhores_params():
    """Lê os hiperparâmetros salvos. Cai no padrão se a busca ainda não rodou."""
    if not PARAMS_JSON.exists():
        return dict(PARAMS_PADRAO)

    dados = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    return dados.get("params", dict(PARAMS_PADRAO))


def avaliar_aninhado(n_trials=25, modo="janela", n_splits=N_SPLITS, mostrar=True):
    """Validação cruzada aninhada: estimativa não enviesada do ganho da busca.

    A busca roda apenas dentro do treino de cada fold externo, e o teste externo
    nunca participa da escolha dos hiperparâmetros.
    """
    _exigir_optuna()
    X, y, groups = montar_dados(modo=modo)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    otimizados, padroes = [], []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # busca interna, cega ao teste externo
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(
            lambda t: _acuracia_cv(espaco_busca(t), X_tr, y_tr, g_tr, n_splits=4)[0],
            n_trials=n_trials,
        )
        params = espaco_busca(optuna.trial.FixedTrial(study.best_params))

        score_otim = accuracy_score(
            y_te, RandomForestClassifier(**params).fit(X_tr, y_tr).predict(X_te)
        )
        score_padrao = accuracy_score(
            y_te, RandomForestClassifier(**PARAMS_PADRAO).fit(X_tr, y_tr).predict(X_te)
        )

        otimizados.append(score_otim)
        padroes.append(score_padrao)

        if mostrar:
            print(f"fold {fold}: otimizado {score_otim:.4f} | padrão {score_padrao:.4f}")

    resultado = {
        "otimizado_media": float(np.mean(otimizados)),
        "otimizado_desvio": float(np.std(otimizados)),
        "padrao_media": float(np.mean(padroes)),
        "padrao_desvio": float(np.std(padroes)),
        "otimizado_folds": [float(s) for s in otimizados],
        "padrao_folds": [float(s) for s in padroes],
    }

    if mostrar:
        print(f"\nAninhada — otimizado: {resultado['otimizado_media']:.4f} "
              f"(±{resultado['otimizado_desvio']:.3f})")
        print(f"Aninhada — padrão:    {resultado['padrao_media']:.4f} "
              f"(±{resultado['padrao_desvio']:.3f})")

    return resultado


if __name__ == "__main__":
    otimizar()
