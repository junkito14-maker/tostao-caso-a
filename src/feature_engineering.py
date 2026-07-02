"""
feature_engineering.py
-----------------------
Construcción de features para un modelo global (pooled) de forecasting de
demanda diaria por tienda-producto.

Enfoque: en vez de ajustar un modelo de series de tiempo por cada una de las
160 combinaciones tienda-producto (con solo 91 días de historia cada una),
se entrena un único modelo de regresión (LightGBM) sobre el panel completo,
usando lags, estadísticas móviles y variables de calendario como features.
Esto permite que el modelo aprenda patrones compartidos entre series
similares, algo crítico dado el historial corto por serie.
"""

from __future__ import annotations

import pandas as pd

LAGS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [7, 14]


def add_calendar_features(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    """Agrega variables de calendario derivadas de la fecha."""
    df = df.copy()
    df["dia_semana"] = df[date_col].dt.dayofweek  # 0=lunes
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["dia_mes"] = df[date_col].dt.day
    df["semana_mes"] = ((df["dia_mes"] - 1) // 7) + 1
    df["mes"] = df[date_col].dt.month
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "unidades_vendidas",
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    """Agrega lags del target por grupo (tienda, producto), ordenado por fecha.

    Importante: el DataFrame debe estar ordenado por fecha dentro de cada
    grupo antes de llamar esta función (se ordena internamente por seguridad).
    """
    df = df.sort_values(["id_tienda", "id_producto", "fecha"]).copy()
    grouped = df.groupby(group_cols)[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "unidades_vendidas",
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Agrega media y desviación móvil del target, calculadas SOLO con
    información pasada (shift(1) antes de la ventana) para evitar leakage.
    """
    df = df.sort_values(["id_tienda", "id_producto", "fecha"]).copy()
    shifted = df.groupby(group_cols)[target_col].shift(1)
    df["_shifted_target"] = shifted
    for window in windows:
        roll = df.groupby(group_cols)["_shifted_target"].rolling(window=window, min_periods=1)
        df[f"rolling_mean_{window}"] = roll.mean().reset_index(level=group_cols, drop=True)
        df[f"rolling_std_{window}"] = roll.std().reset_index(level=group_cols, drop=True)
    df = df.drop(columns=["_shifted_target"])
    return df


def add_store_product_attributes(
    df: pd.DataFrame,
    tiendas: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """Une atributos estáticos de tienda y producto (tamaño, categoría, margen)."""
    df = df.merge(tiendas[["id_tienda", "ciudad", "tamaño_m2"]], on="id_tienda", how="left")
    catalogo_feats = catalogo.copy()
    catalogo_feats["margen_unitario"] = (
        catalogo_feats["precio_venta"] - catalogo_feats["costo_unitario"]
    )
    df = df.merge(
        catalogo_feats[
            [
                "id_producto",
                "categoria",
                "costo_unitario",
                "precio_venta",
                "costo_almacenamiento_semanal",
                "margen_unitario",
            ]
        ],
        on="id_producto",
        how="left",
    )
    return df


def build_feature_table(
    ventas: pd.DataFrame,
    tiendas: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """Pipeline completo de feature engineering sobre el panel diario.

    Devuelve un DataFrame a nivel fecha-tienda-producto, listo para
    entrenar/predecir, con las primeras filas de cada serie (sin historia
    suficiente para lags) incluidas pero con NaNs en esas columnas -- el
    modelo de árboles (LightGBM) maneja NaNs nativamente.
    """
    df = ventas.copy()
    df = add_calendar_features(df)
    df = add_lag_features(df, group_cols=["id_tienda", "id_producto"])
    df = add_rolling_features(df, group_cols=["id_tienda", "id_producto"])
    df = add_store_product_attributes(df, tiendas, catalogo)

    categorical_cols = ["id_tienda", "id_producto", "ciudad", "categoria"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df


FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + [
        "dia_semana",
        "es_fin_de_semana",
        "dia_mes",
        "semana_mes",
        "mes",
        "tamaño_m2",
        "costo_unitario",
        "precio_venta",
        "costo_almacenamiento_semanal",
        "margen_unitario",
    ]
)

CATEGORICAL_FEATURE_COLUMNS = ["id_tienda", "id_producto", "ciudad", "categoria"]

TARGET_COLUMN = "unidades_vendidas"
