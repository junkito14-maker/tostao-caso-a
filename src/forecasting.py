"""
forecasting.py
---------------
Entrenamiento, validación y predicción del modelo de demanda diaria.

Decisión metodológica clave: se usa un modelo global (LightGBM) entrenado
sobre el panel completo de 160 series tienda-producto, en lugar de 160
modelos univariados (Prophet/ETS) independientes. Justificación:

  - Solo hay 91 días (13 semanas) de historia por serie -- insuficiente
    para que un modelo univariado capture estacionalidad de forma robusta.
  - Un modelo pooled aprende patrones compartidos entre series similares
    (mismo producto en distintas tiendas, productos de la misma categoría),
    lo cual mitiga el problema de historia corta.
  - LightGBM maneja NaNs nativamente (lags iniciales) y variables
    categóricas sin necesidad de one-hot encoding.

Validación: Time-Series Split (no K-Fold aleatorio) -- cada fold de
validación usa únicamente fechas posteriores a las de entrenamiento, para
evitar leakage temporal y simular la situación real de producción
(pronosticar el futuro con datos del pasado).

La cuantificación de incertidumbre se hace entrenando 3 modelos con
"quantile" objective (p10, p50, p90), lo cual le da a la capa de
optimización (optimization.py) una distribución aproximada de la demanda
en vez de un único punto estimado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.feature_engineering import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

LGB_PARAMS = {
    "objective": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "verbose": -1,
}


@dataclass
class TimeSeriesSplitResult:
    fold: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    mae: float
    rmse: float
    wape: float


def time_series_splits(
    df: pd.DataFrame, date_col: str = "fecha", n_splits: int = 3, val_days: int = 7
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Genera folds de validación tipo 'walk-forward' (expanding window).

    Cada fold valida sobre los `val_days` posteriores al corte de
    entrenamiento, simulando el pronóstico semanal real. Los folds avanzan
    hacia atrás desde la fecha máxima disponible.
    """
    dates = sorted(df[date_col].unique())
    max_date = dates[-1]

    splits = []
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=val_days * i)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train_end = val_start - pd.Timedelta(days=1)

        train_df = df[df[date_col] <= train_end]
        val_df = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]

        if len(train_df) == 0 or len(val_df) == 0:
            continue
        splits.append((train_df, val_df))

    return list(reversed(splits))  # orden cronológico ascendente


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error -- más robusto que MAPE cuando hay
    valores reales cercanos a cero (común en demanda diaria por SKU-tienda).
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def train_quantile_models(
    train_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    categorical_cols: list[str] = CATEGORICAL_FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
) -> dict[str, lgb.LGBMRegressor]:
    """Entrena un modelo LightGBM por cuantil (p10, p50, p90)."""
    all_features = feature_cols + categorical_cols
    X_train = train_df[all_features]
    y_train = train_df[target_col]

    models: dict[str, lgb.LGBMRegressor] = {}
    for name, alpha in QUANTILES.items():
        model = lgb.LGBMRegressor(**LGB_PARAMS, alpha=alpha)
        model.fit(X_train, y_train, categorical_feature=categorical_cols)
        models[name] = model
    return models


def predict_quantiles(
    models: dict[str, lgb.LGBMRegressor],
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    categorical_cols: list[str] = CATEGORICAL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Genera predicciones p10/p50/p90 y las ordena (p10 <= p50 <= p90)."""
    all_features = feature_cols + categorical_cols
    X = df[all_features]

    preds = pd.DataFrame(index=df.index)
    for name, model in models.items():
        preds[name] = model.predict(X)

    # Garantizar monotonicidad entre cuantiles (LightGBM no lo asegura por defecto)
    preds_sorted = np.sort(preds[["p10", "p50", "p90"]].to_numpy(), axis=1)
    preds["p10"], preds["p50"], preds["p90"] = (
        preds_sorted[:, 0],
        preds_sorted[:, 1],
        preds_sorted[:, 2],
    )
    preds[["p10", "p50", "p90"]] = preds[["p10", "p50", "p90"]].clip(lower=0)
    return preds


def forecast_next_week(
    models: dict[str, lgb.LGBMRegressor],
    historical_df: pd.DataFrame,
    tiendas: pd.DataFrame,
    catalogo: pd.DataFrame,
    n_days: int = 7,
) -> pd.DataFrame:
    """Pronostica la demanda agregada de la siguiente semana (n_days) por
    serie tienda-producto, mediante forecasting recursivo (multi-step).

    Cómo funciona: para cada uno de los 7 días futuros, se recalculan los
    features (lags, rolling stats) usando las predicciones de los días
    anteriores como si fueran ventas reales ("recursive strategy"), y se
    vuelve a invocar el modelo. Esto se hace de forma independiente para
    cada cuantil (p10, p50, p90), y al final se suman los 7 valores diarios
    de cada cuantil para obtener la demanda semanal total.

    Limitación reconocida: sumar cuantiles diarios pronosticados de forma
    independiente subestima levemente la varianza real acumulada de 7 días
    (no captura la correlación serial completa). Es una aproximación
    razonable y transparente para el alcance de este ejercicio; una mejora
    futura sería modelar directamente la demanda semanal agregada o usar
    simulación conjunta (Monte Carlo) sobre la trayectoria completa.
    """
    from src.feature_engineering import (
        add_calendar_features,
        add_lag_features,
        add_rolling_features,
        add_store_product_attributes,
    )

    weekly_totals = {name: None for name in models}

    for name, model in models.items():
        working_df = historical_df[["fecha", "id_tienda", "id_producto", TARGET_COLUMN]].copy()
        last_date = working_df["fecha"].max()
        daily_preds = []

        for step in range(1, n_days + 1):
            future_date = last_date + pd.Timedelta(days=step)
            series_ids = working_df[["id_tienda", "id_producto"]].drop_duplicates()
            future_rows = series_ids.copy()
            future_rows["fecha"] = future_date
            future_rows[TARGET_COLUMN] = np.nan

            extended = pd.concat([working_df, future_rows], ignore_index=True)
            extended = add_calendar_features(extended)
            extended = add_lag_features(extended, group_cols=["id_tienda", "id_producto"])
            extended = add_rolling_features(extended, group_cols=["id_tienda", "id_producto"])
            extended = add_store_product_attributes(extended, tiendas, catalogo)
            for col in CATEGORICAL_FEATURE_COLUMNS:
                extended[col] = extended[col].astype("category")

            target_rows = extended[extended["fecha"] == future_date].copy()
            X_future = target_rows[FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
            preds = model.predict(X_future)
            preds = np.clip(preds, 0, None)

            target_rows[TARGET_COLUMN] = preds
            daily_preds.append(
                target_rows[["id_tienda", "id_producto", "fecha", TARGET_COLUMN]].copy()
            )

            # Realimentar la predicción como "histórico" para el siguiente paso recursivo
            working_df = pd.concat(
                [working_df, target_rows[["fecha", "id_tienda", "id_producto", TARGET_COLUMN]]],
                ignore_index=True,
            )

        all_days = pd.concat(daily_preds, ignore_index=True)
        weekly_sum = (
            all_days.groupby(["id_tienda", "id_producto"])[TARGET_COLUMN]
            .sum()
            .rename(name)
            .reset_index()
        )
        weekly_totals[name] = weekly_sum

    result = weekly_totals["p10"]
    for name in ["p50", "p90"]:
        result = result.merge(weekly_totals[name], on=["id_tienda", "id_producto"])

    # Garantizar monotonicidad p10 <= p50 <= p90 también a nivel semanal
    sorted_vals = np.sort(result[["p10", "p50", "p90"]].to_numpy(), axis=1)
    result["p10"], result["p50"], result["p90"] = (
        sorted_vals[:, 0],
        sorted_vals[:, 1],
        sorted_vals[:, 2],
    )
    return result


def run_time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 7,
) -> list[TimeSeriesSplitResult]:
    """Corre walk-forward validation y devuelve métricas por fold (sobre p50)."""
    splits = time_series_splits(df, n_splits=n_splits, val_days=val_days)
    results = []

    for i, (train_df, val_df) in enumerate(splits):
        models = train_quantile_models(train_df)
        preds = predict_quantiles(models, val_df)

        y_true = val_df[TARGET_COLUMN].to_numpy()
        y_pred = preds["p50"].to_numpy()

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        wape = _wape(y_true, y_pred)

        results.append(
            TimeSeriesSplitResult(
                fold=i,
                train_end=train_df["fecha"].max(),
                val_start=val_df["fecha"].min(),
                val_end=val_df["fecha"].max(),
                mae=mae,
                rmse=rmse,
                wape=wape,
            )
        )
        logger.info(
            "Fold %d | train hasta %s | val %s -> %s | MAE=%.2f RMSE=%.2f WAPE=%.2f%%",
            i,
            train_df["fecha"].max().date(),
            val_df["fecha"].min().date(),
            val_df["fecha"].max().date(),
            mae,
            rmse,
            wape * 100,
        )

    return results
