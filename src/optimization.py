"""
optimization.py
-----------------
Función de optimización de cantidad de pedido por SKU-tienda, balanceando
el costo esperado de Stockout (quedarse corto) vs. Overstock (pedir de más).

Enfoque: modelo "Newsvendor" (fractil crítico), el estándar de la industria
para decisiones de inventario bajo demanda incierta de un solo periodo:

    fractil_critico = Cu / (Cu + Co)

donde:
    Cu = costo unitario de Underage (stockout) = margen perdido por unidad
         no vendida = precio_venta - costo_unitario
    Co = costo unitario de Overage (overstock) = costo de mantener una
         unidad de inventario sin vender = costo_almacenamiento_semanal

La cantidad óptima a pedir Q* es el cuantil de la distribución de demanda
que corresponde al fractil crítico. Usamos los cuantiles p10/p50/p90 que
entrega el modelo de forecasting (forecasting.py) para aproximar la
distribución de demanda mediante una Normal (loc=p50, scale derivado del
spread p10-p90), y de ahí interpolamos el cuantil exacto que necesitamos.

Esto conecta directamente la incertidumbre del modelo (intervalos de
confianza) con el margen del producto: productos de margen alto justifican
pedidos más agresivos (fractil crítico alto, cerca de p90); productos de
margen bajo y alto costo de almacenamiento justifican pedidos conservadores
(fractil crítico bajo, cerca de p10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# z-score correspondiente al percentil 90 de una Normal estándar
_Z_P90 = stats.norm.ppf(0.90)


def critical_fractile(margen_unitario: pd.Series, costo_almacenamiento: pd.Series) -> pd.Series:
    """Calcula el fractil crítico (Cu / (Cu+Co)) por fila.

    Cu = margen_unitario (costo de oportunidad de no vender una unidad)
    Co = costo_almacenamiento_semanal (costo de quedarse con una unidad sin vender)
    """
    cu = margen_unitario.clip(lower=0.01)  # evitar división por cero
    co = costo_almacenamiento.clip(lower=0.01)
    return cu / (cu + co)


def _implied_std(p10: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """Estima la desviación estándar de la demanda a partir del spread p10-p90,
    asumiendo aproximación Normal. std = (p90 - p10) / (2 * z_p90).
    """
    std = (p90 - p10) / (2 * _Z_P90)
    return np.clip(std, a_min=0.5, a_max=None)  # piso mínimo para evitar std=0


def optimal_order_quantity(
    forecast_df: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula la cantidad óptima de pedido Q* por fila (tienda-producto-fecha).

    Parameters
    ----------
    forecast_df : DataFrame con columnas id_tienda, id_producto, p10, p50, p90
    catalogo : DataFrame con costo_unitario, precio_venta, costo_almacenamiento_semanal

    Returns
    -------
    DataFrame con columnas adicionales: fractil_critico, demanda_std,
    cantidad_optima_pedido (Q*, sin redondear hacia arriba todavía).
    """
    df = forecast_df.merge(
        catalogo[["id_producto", "costo_unitario", "precio_venta", "costo_almacenamiento_semanal"]],
        on="id_producto",
        how="left",
    )
    df["margen_unitario"] = df["precio_venta"] - df["costo_unitario"]
    df["fractil_critico"] = critical_fractile(
        df["margen_unitario"], df["costo_almacenamiento_semanal"]
    )
    df["demanda_std"] = _implied_std(df["p10"].to_numpy(), df["p90"].to_numpy())

    df["cantidad_optima_pedido"] = stats.norm.ppf(
        df["fractil_critico"], loc=df["p50"], scale=df["demanda_std"]
    )
    df["cantidad_optima_pedido"] = df["cantidad_optima_pedido"].clip(lower=0)
    return df


def order_recommendation(
    forecast_with_optimal: pd.DataFrame,
    inventario: pd.DataFrame,
) -> pd.DataFrame:
    """Resta el stock actual a la cantidad óptima para obtener la recomendación
    final de cuánto pedir (no se puede pedir cantidades negativas).
    """
    df = forecast_with_optimal.merge(
        inventario[["id_tienda", "id_producto", "stock_actual"]],
        on=["id_tienda", "id_producto"],
        how="left",
    )
    df["pedido_recomendado"] = np.ceil(
        (df["cantidad_optima_pedido"] - df["stock_actual"]).clip(lower=0)
    ).astype(int)
    return df


def expected_cost_comparison(
    df: pd.DataFrame,
    naive_policy_col: str = "p50",
    optimal_policy_col: str = "cantidad_optima_pedido",
) -> pd.DataFrame:
    """Compara el costo esperado de la política naive (pedir = pronóstico p50)
    vs. la política óptima (newsvendor), usando simulación Monte Carlo simple
    sobre la distribución Normal implícita de cada serie.

    Devuelve el DataFrame con columnas de costo esperado para ambas políticas,
    útil para cuantificar el ahorro de negocio en la presentación ejecutiva.
    """
    rng = np.random.default_rng(42)
    n_sims = 500

    costos_naive = []
    costos_optimo = []

    for _, row in df.iterrows():
        demanda_sim = rng.normal(row["p50"], row["demanda_std"], size=n_sims)
        demanda_sim = np.clip(demanda_sim, 0, None)

        cu = row["margen_unitario"]
        co = row["costo_almacenamiento_semanal"]

        for policy_col, costos_list in [
            (naive_policy_col, costos_naive),
            (optimal_policy_col, costos_optimo),
        ]:
            q = row[policy_col]
            stockout_units = np.clip(demanda_sim - q, 0, None)
            overstock_units = np.clip(q - demanda_sim, 0, None)
            costo_esperado = float(np.mean(stockout_units * cu + overstock_units * co))
            costos_list.append(costo_esperado)

    df = df.copy()
    df["costo_esperado_naive"] = costos_naive
    df["costo_esperado_optimo"] = costos_optimo
    df["ahorro_estimado"] = df["costo_esperado_naive"] - df["costo_esperado_optimo"]
    return df
