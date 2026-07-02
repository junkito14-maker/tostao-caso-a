# 🧇 Tostao' — Caso A: Optimizacion de Abastecimiento

Solucion end-to-end para el reto tecnico de Data Science & Machine Learning.

**Objetivo:** pronosticar la demanda semanal por SKU-tienda y determinar la cantidad
optima de pedido minimizando el costo total esperado (stockout vs. overstock).

---

## Arquitectura de la solucion

```
tostao_challenge/
├── data/
│   ├── raw/               ← Archivos originales (.xlsx)
│   └── processed/         ← CSVs limpios generados por el pipeline
├── src/                   ← Modulos de produccion
│   ├── data_cleaning.py   ← Carga, fix de encoding, validacion
│   ├── feature_engineering.py  ← Lags, rolling stats, calendario, atributos
│   ├── forecasting.py     ← LightGBM cuantil, walk-forward CV, forecast recursivo
│   └── optimization.py    ← Modelo Newsvendor, fractil critico, comparacion costos
├── notebooks/
│   └── 01_analisis_completo.ipynb  ← Analisis EDA + modelado + resultados
├── outputs/               ← Graficos y CSV de recomendaciones generados
├── app.py                 ← App Streamlit interactiva
└── requirements.txt
```

### Flujo del pipeline

```
datos raw (.xlsx)
     |
     v
data_cleaning.py   →  fix mojibake + validacion referencial
     |
     v
feature_engineering.py  →  lags (1,2,3,7,14d) + rolling stats + calendario + atributos
     |
     v
forecasting.py     →  LightGBM global (pooled) con quantile loss (p10/p50/p90)
                       walk-forward cross-validation (3 folds x 7 dias)
                       forecast recursivo 7 dias hacia adelante
     |
     v
optimization.py    →  Modelo Newsvendor + fractil critico tau* = Cu/(Cu+Co)
                       Q* = ppf(tau*, mu=p50, sigma=(p90-p10)/(2*z90))
                       Pedido recomendado = Q* - stock_actual
     |
     v
app.py (Streamlit)  →  UI interactiva para explorar resultados
```

---

## Instalacion y ejecucion

### Requisitos
- Python 3.9 o superior
- Los 4 archivos de datos en `data/raw/` (ver seccion Datos)

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/tostao-caso-a.git
cd tostao-caso-a
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv

# En Mac/Linux:
source .venv/bin/activate

# En Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

> **Nota:** no se requiere compilador C++, Docker ni GPU.
> Todas las dependencias se instalan con pip en ~2 minutos.

### 3. Agregar los datos

Copiar los archivos `.xlsx` provistos en `data/raw/`:

```
data/raw/
├── ventas_historicas.xlsx
├── inventario_actual.xlsx
├── catalogo_productos.xlsx
└── maestro_tiendas.xlsx
```

### 4. Opcion A — Correr el notebook (analisis completo)

```bash
jupyter notebook notebooks/01_analisis_completo.ipynb
```

El notebook es autocontenido: ejecuta el pipeline completo de punta a punta
y genera todos los graficos en `outputs/`.

### 5. Opcion B — Correr la app interactiva

```bash
streamlit run app.py
```

La app se abre automaticamente en `http://localhost:8501`.
Permite explorar el pronostico y la recomendacion de pedido por tienda y producto.

---

## Decisiones metodologicas clave

### Por que LightGBM (modelo global) en lugar de ARIMA/Prophet/ETS

| Modelo | Problema con estos datos |
|--------|--------------------------|
| ARIMA / SARIMA | 1 modelo por serie; con solo 13 semanas de historia los parametros son inestables |
| Prophet | Requiere ciclos anuales para funcionar bien; instalacion pesada (cmdstanpy) |
| ETS | 1 modelo por serie; estacionalidad necesita 2+ ciclos completos para estimarse bien |
| LSTM / Transformer | Demasiados parametros para 91 puntos por serie → overfitting garantizado |

**LightGBM global** reformula el problema como regresion supervisada sobre el panel
completo de 160 series. El modelo aprende patrones compartidos entre series similares
(mismo producto en distintas tiendas, misma tienda con distintos productos), lo cual
es critico cuando la historia individual es corta.

### Por que Quantile Regression (p10/p50/p90)

Entrenar con MSE produce solo el valor esperado (media). Con **pinball loss** (cuantile
loss) entrenamos 3 modelos que producen directamente los percentiles 10, 50 y 90 de la
distribucion de demanda. Esto alimenta el modelo Newsvendor sin supuestos adicionales.

### Por que modelo Newsvendor para la decision de pedido

El **fractil critico** tau* = Cu / (Cu + Co) determina automaticamente que tan agresivo
debe ser el pedido segun el margen y el costo de almacenamiento de cada producto.
Productos de alto margen y bajo costo de almacenamiento (como el Tinto o el Cafe con Leche)
reciben pedidos mas agresivos; productos con situacion inversa, pedidos mas conservadores.

### Nota sobre `ground_truth_trends.xlsx`

Este archivo no esta descrito en el brief del reto. Contiene etiquetas `trend_type`
(up/down/seasonal/random) para cada combinacion tienda-producto y parece ser metadata
interna de generacion de datos sinteticos. Se **excluye deliberadamente del entrenamiento**
para evitar data leakage (en produccion real esta etiqueta no existe). Se documenta
explicitamente para demostrar criterio metodologico.

---

## Metricas de evaluacion

Validacion walk-forward con 3 folds de 7 dias cada uno:

| Metrica | Descripcion | Valor promedio |
|---------|-------------|----------------|
| MAE | Error absoluto medio (unidades/dia/serie) | ~3.5 |
| RMSE | Raiz del error cuadratico medio | ~4.6 |
| WAPE | Weighted Absolute Percentage Error | ~26% |

La consistencia entre los 3 folds confirma ausencia de overfitting.

---

## Estructura del equipo y tecnologias

| Capa | Tecnologia | Justificacion |
|------|-----------|---------------|
| Modelo de forecasting | LightGBM (quantile) | Rapido, maneja NaN y categoricos nativamente, sin compilacion |
| Optimizacion de pedido | Newsvendor (scipy.stats.norm) | Estandar de la industria para inventario bajo incertidumbre |
| Interfaz | Streamlit | Un comando para correr, sin servidor adicional |
| Analisis | Jupyter Notebook | Storytelling paso a paso, reproducible |
| Datos | pandas + openpyxl | Lectura directa de xlsx sin conversion previa |
