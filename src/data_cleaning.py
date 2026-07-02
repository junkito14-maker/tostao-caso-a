"""
data_cleaning.py
-----------------
Carga, limpieza y validación de los datasets crudos del reto de Optimización
de Abastecimiento (Caso A).

Responsabilidades:
    1. Leer los archivos Excel crudos (data/raw/).
    2. Corregir problemas de encoding (mojibake UTF-8 leído como Latin-1).
    3. Validar integridad referencial entre tablas.
    4. Validar reglas de negocio (no negativos, grilla completa de fechas, etc.).
    5. Persistir versiones limpias en data/processed/ como CSV.

Nota sobre `ground_truth_trends.xlsx`:
    Este archivo no forma parte de los datasets descritos en el brief del
    reto. Contiene la etiqueta `trend_type` (up/down/seasonal/random) que
    parece ser metadata interna usada para generar los datos sintéticos de
    ventas. Deliberadamente NO se usa como insumo de entrenamiento del
    modelo (evitar data leakage / no replicable en producción real). Se
    carga únicamente para fines de validación exploratoria post-hoc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass
class RetailDataset:
    """Contenedor tipado para las tablas limpias del reto."""

    ventas: pd.DataFrame
    tiendas: pd.DataFrame
    inventario: pd.DataFrame
    catalogo: pd.DataFrame
    ground_truth: pd.DataFrame | None = None


def _fix_mojibake(value: object) -> object:
    """Corrige strings con doble-encoding (UTF-8 bytes leídos como Latin-1).

    Ejemplo: 'CafÃ©' -> 'Café'. Si el string no tiene el problema, lo
    devuelve sin cambios (best-effort, no lanza excepción).
    """
    if not isinstance(value, str):
        return value
    try:
        fixed = value.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return value
    return fixed


def _fix_mojibake_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica _fix_mojibake a nombres de columnas y a todas las columnas tipo string."""
    df = df.rename(columns={c: _fix_mojibake(c) for c in df.columns})
    string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"]
    for col in string_cols:
        df[col] = df[col].map(_fix_mojibake)
    return df


def load_raw_tables(raw_dir: Path = RAW_DIR) -> RetailDataset:
    """Lee los 5 archivos xlsx crudos y aplica la corrección de encoding."""
    logger.info("Leyendo archivos crudos desde %s", raw_dir)

    ventas = pd.read_excel(raw_dir / "ventas_historicas.xlsx")
    tiendas = pd.read_excel(raw_dir / "maestro_tiendas.xlsx")
    inventario = pd.read_excel(raw_dir / "inventario_actual.xlsx")
    catalogo = pd.read_excel(raw_dir / "catalogo_productos.xlsx")
    ground_truth = pd.read_excel(raw_dir / "ground_truth_trends.xlsx")

    ventas = _fix_mojibake_columns(ventas)
    tiendas = _fix_mojibake_columns(tiendas)
    inventario = _fix_mojibake_columns(inventario)
    catalogo = _fix_mojibake_columns(catalogo)
    ground_truth = _fix_mojibake_columns(ground_truth)

    ventas["fecha"] = pd.to_datetime(ventas["fecha"])

    return RetailDataset(
        ventas=ventas,
        tiendas=tiendas,
        inventario=inventario,
        catalogo=catalogo,
        ground_truth=ground_truth,
    )


def validate_dataset(data: RetailDataset) -> list[str]:
    """Corre validaciones de calidad e integridad. Devuelve lista de advertencias.

    No lanza excepción salvo que se rompa una regla crítica (ej. negativos
    en ventas o stock), en cuyo caso sí levanta ValueError.
    """
    warnings: list[str] = []

    # --- Reglas críticas (rompen el pipeline si fallan) ---
    if (data.ventas["unidades_vendidas"] < 0).any():
        raise ValueError("Se encontraron unidades_vendidas negativas.")
    if (data.inventario["stock_actual"] < 0).any():
        raise ValueError("Se encontraron valores de stock_actual negativos.")

    # --- Integridad referencial ---
    tiendas_ventas = set(data.ventas["id_tienda"]) - set(data.tiendas["id_tienda"])
    if tiendas_ventas:
        warnings.append(f"Tiendas en ventas sin maestro: {tiendas_ventas}")

    productos_ventas = set(data.ventas["id_producto"]) - set(data.catalogo["id_producto"])
    if productos_ventas:
        warnings.append(f"Productos en ventas sin catálogo: {productos_ventas}")

    # --- Completitud de la grilla tienda x producto x fecha ---
    n_tiendas = data.ventas["id_tienda"].nunique()
    n_productos = data.ventas["id_producto"].nunique()
    n_fechas = data.ventas["fecha"].nunique()
    filas_esperadas = n_tiendas * n_productos * n_fechas
    if len(data.ventas) != filas_esperadas:
        warnings.append(
            f"Grilla incompleta: se esperaban {filas_esperadas} filas, hay {len(data.ventas)}."
        )

    duplicados = data.ventas.duplicated(subset=["fecha", "id_tienda", "id_producto"]).sum()
    if duplicados:
        warnings.append(f"{duplicados} filas duplicadas en ventas_historicas.")

    for warning in warnings:
        logger.warning(warning)
    if not warnings:
        logger.info("Validación completada sin advertencias.")

    return warnings


def save_processed(data: RetailDataset, processed_dir: Path = PROCESSED_DIR) -> None:
    """Persiste las tablas limpias como CSV en data/processed/."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    data.ventas.to_csv(processed_dir / "ventas_historicas.csv", index=False)
    data.tiendas.to_csv(processed_dir / "maestro_tiendas.csv", index=False)
    data.inventario.to_csv(processed_dir / "inventario_actual.csv", index=False)
    data.catalogo.to_csv(processed_dir / "catalogo_productos.csv", index=False)
    if data.ground_truth is not None:
        data.ground_truth.to_csv(processed_dir / "ground_truth_trends.csv", index=False)
    logger.info("Datos procesados guardados en %s", processed_dir)


def run_pipeline(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> RetailDataset:
    """Orquesta carga -> limpieza -> validación -> persistencia."""
    data = load_raw_tables(raw_dir)
    validate_dataset(data)
    save_processed(data, processed_dir)
    return data


if __name__ == "__main__":
    run_pipeline()
