"""
app.py — Tostao' | Optimizacion de Abastecimiento
Para correr: streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_cleaning import load_raw_tables
from src.feature_engineering import build_feature_table
from src.forecasting import forecast_next_week, run_time_series_cv, train_quantile_models
from src.optimization import expected_cost_comparison, optimal_order_quantity, order_recommendation

BROWN = "#4E2C0E"
GOLD  = "#C8942A"
CREAM = "#FAF3E0"
OLIVE = "#7A8C50"
RED   = "#C0392B"
WHITE = "#FFFFFF"

st.set_page_config(
    page_title="Tostao' | Abastecimiento",
    page_icon="🧇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .stApp {{ background-color: {CREAM}; }}

  /* ── Sidebar fondo ── */
  section[data-testid="stSidebar"] {{ background-color: {BROWN}; }}

  /* ── Labels en dorado ── */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label {{
      color: {GOLD} !important;
      font-weight: 700 !important;
      font-size: 0.8rem !important;
      letter-spacing: 0.06em !important;
  }}

  /* ── FIX texto visible en selectbox:
       forzamos CAFE OSCURO en todos los nodos de texto dentro del widget ── */
  section[data-testid="stSidebar"] div[data-baseweb="select"] {{
      background-color: {WHITE} !important;
      border-radius: 6px !important;
  }}
  section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
      color: {BROWN} !important;
      background-color: transparent !important;
  }}
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
      background-color: {WHITE} !important;
  }}

  /* ── KPI cards ── */
  div[data-testid="metric-container"] {{
      background: {WHITE};
      border-left: 4px solid {GOLD};
      border-radius: 8px;
      padding: 14px 18px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }}
  div[data-testid="metric-container"] label {{
      color: {BROWN} !important; font-size: 0.75rem !important;
      font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
      color: {BROWN} !important; font-size: 1.8rem !important; font-weight: 700 !important;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {{
      font-size: 0.8rem !important;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{ border-bottom: 2px solid {GOLD}; gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{ color: {BROWN}; font-weight: 600; padding: 8px 20px; }}
  .stTabs [aria-selected="true"] {{ color: {GOLD} !important; border-bottom: 3px solid {GOLD}; }}

  /* ── Titulos ── */
  h1, h2, h3 {{ color: {BROWN} !important; }}

  /* ── Alertas ── */
  .alert-red    {{ background:#FADBD8; border-left:4px solid {RED};  border-radius:6px; padding:10px 14px; color:{RED};    font-weight:600; margin:6px 0; }}
  .alert-yellow {{ background:#FEF9E7; border-left:4px solid {GOLD}; border-radius:6px; padding:10px 14px; color:#7D6608; font-weight:600; margin:6px 0; }}
  .alert-green  {{ background:#D5F5E3; border-left:4px solid {OLIVE};border-radius:6px; padding:10px 14px; color:#1E8449; font-weight:600; margin:6px 0; }}

  /* ── Info box (reemplaza blockquote) ── */
  .info-box {{
      background: {WHITE}; border-left: 3px solid {GOLD};
      border-radius: 6px; padding: 12px 16px;
      color: {BROWN}; font-size: 0.88rem; line-height: 1.6;
      margin: 8px 0; font-family: 'DejaVu Sans', sans-serif;
  }}
  .info-box b {{ color: {BROWN}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_train():
    data   = load_raw_tables()
    feat   = build_feature_table(data.ventas, data.tiendas, data.catalogo)
    models = train_quantile_models(feat)
    weekly = forecast_next_week(models, data.ventas, data.tiendas, data.catalogo, n_days=7)
    opt    = optimal_order_quantity(weekly, data.catalogo)
    rec    = order_recommendation(opt, data.inventario)
    res    = expected_cost_comparison(rec)
    tcol   = [c for c in data.tiendas.columns if "tama" in c.lower()][0]
    res    = (res
        .merge(data.catalogo[["id_producto","nombre","categoria"]], on="id_producto")
        .merge(data.tiendas[["id_tienda","ciudad", tcol]], on="id_tienda")
        .rename(columns={tcol: "tamano_m2"})
    )
    return data, feat, models, res

@st.cache_data(show_spinner=False)
def run_cv(_feat):
    return run_time_series_cv(_feat, n_splits=3, val_days=7)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS GRAFICOS
# ══════════════════════════════════════════════════════════════════════════════

def mpl_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def chart_stores_overview(res_prod, ventas, producto_sel):
    dem = (ventas[ventas["id_producto"] == producto_sel]
           .groupby("id_tienda")["unidades_vendidas"].mean().reset_index())
    dem.columns = ["id_tienda", "dem_diaria"]
    df = res_prod.merge(dem, on="id_tienda").sort_values("p50", ascending=True)
    df["dias_cob"] = df["stock_actual"] / df["dem_diaria"].clip(lower=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(CREAM)

    for ax in axes:
        ax.set_facecolor(CREAM)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    y = range(len(df))

    # ── Izquierda: demanda vs stock ───────────────────────────────────────
    ax = axes[0]
    ax.barh(y, df["p90"], color=GOLD, alpha=0.18)
    ax.barh(y, df["p10"], color=CREAM, alpha=1.0)
    ax.barh(y, df["p50"], color=GOLD, alpha=0.85, label="Demanda esperada p50")

    for i, (_, row) in enumerate(df.iterrows()):
        col = RED if row["stock_actual"] < row["p10"] else OLIVE
        ax.plot(row["stock_actual"], i, "D", color=col, markersize=7, zorder=5)

    # FIX: etiquetas en una sola linea, fuente pequeña
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [f"{r.id_tienda} · {r.ciudad}" for _, r in df.iterrows()],
        fontsize=7.5
    )
    ax.set_xlabel("Unidades / semana")
    ax.set_title(
        "Demanda esperada (barra) vs. Stock actual (diamante)\n"
        "Rojo = stock por debajo del escenario pesimista (p10)",
        color=BROWN, fontweight="bold", fontsize=10, pad=10
    )

    from matplotlib.lines import Line2D
    legend_el = [
        plt.Rectangle((0,0),1,1, color=GOLD, alpha=0.85, label="Demanda p50"),
        plt.Rectangle((0,0),1,1, color=GOLD, alpha=0.18, label="Intervalo p10-p90"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=RED,   markersize=7, label="Stock critico"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=OLIVE, markersize=7, label="Stock OK"),
    ]
    # Leyenda FUERA del area del plot (abajo)
    ax.legend(handles=legend_el, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    # ── Derecha: pedido recomendado ───────────────────────────────────────
    ax2 = axes[1]
    colors_p = [RED if r.pedido_recomendado > r.p50 * 0.4 else OLIVE for _, r in df.iterrows()]
    ax2.barh(y, df["pedido_recomendado"], color=colors_p, alpha=0.8, edgecolor=WHITE)

    for i, (_, row) in enumerate(df.iterrows()):
        if row["pedido_recomendado"] > 0:
            ax2.text(row["pedido_recomendado"] + 0.5, i,
                     f'{int(row["pedido_recomendado"])} uds',
                     va="center", fontsize=7.5, color=BROWN)

    ax2.set_yticks(list(y))
    ax2.set_yticklabels(
        [f"{r.id_tienda} · {r.ciudad}" for _, r in df.iterrows()],
        fontsize=7.5
    )
    ax2.set_xlabel("Unidades a pedir")
    ax2.set_title(
        "Pedido recomendado por tienda\nRojo = pedido grande · Verde = stock suficiente",
        color=BROWN, fontweight="bold", fontsize=10, pad=10
    )

    fig.tight_layout(pad=2.5, rect=[0, 0.05, 1, 1])
    return fig, df


def chart_heatmap_cobertura(ventas, inventario, tiendas, catalogo):
    import seaborn as sns
    inv = inventario.merge(catalogo[["id_producto","nombre"]], on="id_producto")
    inv = inv.merge(tiendas[["id_tienda","ciudad"]], on="id_tienda")
    dem = (ventas.groupby(["id_tienda","id_producto"])["unidades_vendidas"]
           .mean().reset_index())
    dem.columns = ["id_tienda","id_producto","dem_diaria"]
    inv = inv.merge(dem, on=["id_tienda","id_producto"])
    inv["dias_cob"] = inv["stock_actual"] / inv["dem_diaria"].clip(lower=0.01)
    pivot = inv.pivot(index="id_tienda", columns="nombre", values="dias_cob")

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=5, vmin=0, vmax=10,
                annot=True, fmt=".1f", linewidths=0.5, linecolor=WHITE,
                cbar_kws={"label":"Dias de cobertura","shrink":0.7})
    ax.set_title(
        "Dias de cobertura de inventario · Todas las tiendas × Todos los productos\n"
        "Rojo < 2 dias = riesgo critico   |   Verde > 5 dias = OK",
        color=BROWN, fontweight="bold", fontsize=11
    )
    ax.set_xlabel("Producto")
    ax.set_ylabel("Tienda")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return fig


def chart_forecast_serie(ventas, tienda, producto, row_fc, nombre_prod):
    hist = (ventas[(ventas["id_tienda"]==tienda) & (ventas["id_producto"]==producto)]
            .sort_values("fecha"))
    ma7  = hist["unidades_vendidas"].rolling(7, min_periods=1).mean()

    last_date    = hist["fecha"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)

    # Convertir semanal -> diario para misma escala que historico
    p50_d = row_fc["p50"] / 7
    p10_d = row_fc["p10"] / 7
    p90_d = row_fc["p90"] / 7
    last_v = float(hist["unidades_vendidas"].iloc[-1])

    fig, ax = mpl_fig((10, 3.5))
    ax.fill_between(hist["fecha"], hist["unidades_vendidas"], alpha=0.15, color=BROWN)
    ax.plot(hist["fecha"], hist["unidades_vendidas"], color=BROWN, lw=0.9, alpha=0.5, label="Ventas reales")
    ax.plot(hist["fecha"], ma7, color=BROWN, lw=2, label="Media movil 7d")
    ax.fill_between(future_dates,
                    np.linspace(last_v, p10_d, 7),
                    np.linspace(last_v, p90_d, 7),
                    alpha=0.3, color=GOLD, label="Intervalo p10-p90")
    ax.plot(future_dates, np.linspace(last_v, p50_d, 7),
            color=GOLD, lw=2.5, linestyle="--",
            label=f"Pronostico p50 (~{p50_d:.0f} uds/dia)")
    ax.axvline(last_date, color="gray", lw=1, linestyle=":", alpha=0.6)
    ax.text(last_date, ax.get_ylim()[1] * 0.95, "  hoy", color="gray", fontsize=8)
    ax.set_title(f"{tienda} — {nombre_prod} | unidades por dia",
                 color=BROWN, fontweight="bold", fontsize=11)
    ax.set_ylabel("Unidades / dia")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    fig.tight_layout()
    return fig


def chart_newsvendor(row):
    from scipy.stats import norm
    mu_w    = row["p50"]
    sigma_w = max((row["p90"] - row["p10"]) / (2 * 1.2816), 0.5)
    tau     = row["fractil_critico"]
    q_opt   = row["cantidad_optima_pedido"]
    stock   = row["stock_actual"]

    x = np.linspace(max(0, mu_w - 4*sigma_w), mu_w + 4*sigma_w, 400)
    y_pdf = norm.pdf(x, mu_w, sigma_w)

    # FIX: figura mas ancha para que la leyenda quepa afuera
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.fill_between(x, y_pdf, alpha=0.15, color=BROWN)
    ax.plot(x, y_pdf, color=BROWN, lw=2)

    x_so = x[x >= q_opt]
    ax.fill_between(x_so, norm.pdf(x_so, mu_w, sigma_w),
                    alpha=0.4, color=RED,
                    label=f"Riesgo stockout ({(1-tau)*100:.1f}%)")
    ax.axvline(q_opt,  color=GOLD,  lw=2.5, linestyle="--",
               label=f"Q* optimo = {q_opt:.0f} uds/sem")
    ax.axvline(stock,  color=OLIVE, lw=2,   linestyle=":",
               label=f"Stock actual = {stock:.0f} uds")
    ax.axvline(mu_w,   color=BROWN, lw=1.5, linestyle="-", alpha=0.5,
               label=f"Demanda esperada = {mu_w:.0f} uds/sem")

    ax.set_xlabel("Unidades demandadas (semana)")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title("Distribucion de demanda semanal y decision Newsvendor",
                 color=BROWN, fontweight="bold", fontsize=10)

    # FIX: leyenda completamente FUERA del area del grafico (derecha)
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True,
              framealpha=0.95, edgecolor=GOLD)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    return fig


def chart_cost_bars(row):
    fig, ax = mpl_fig((5, 3))
    labels = ["Naive\n(pedir p50)", "Optimo\n(Newsvendor)"]
    values = [row["costo_esperado_naive"], row["costo_esperado_optimo"]]
    bars   = ax.bar(labels, values, color=[RED, OLIVE], alpha=0.85, edgecolor=WHITE, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f"${val:,.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=BROWN)
    ax.set_ylabel("Costo esperado ($COP)")
    ax.set_title("Costo por politica de pedido", color=BROWN, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CARGA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Cargando datos y entrenando modelo..."):
    data, feat, models, results = load_and_train()

nombres_map = dict(zip(data.catalogo["id_producto"], data.catalogo["nombre"]))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 10px 0;'>
      <div style='font-size:2.5rem;'>🧇</div>
      <div style='color:{GOLD}; font-size:1.3rem; font-weight:700; letter-spacing:0.05em;'>TOSTAO'</div>
      <div style='color:#D5C5B0; font-size:0.75rem; margin-top:4px;'>Optimizacion de Abastecimiento</div>
    </div>
    <hr style='border-color:{GOLD}; opacity:0.3; margin:10px 0 18px 0;'>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{GOLD}; font-size:0.78rem; font-weight:700; "
        f"letter-spacing:0.08em; margin-bottom:6px;'>PRODUCTO</p>",
        unsafe_allow_html=True
    )

    producto_sel = st.selectbox(
        "Producto",
        sorted(results["id_producto"].unique()),
        format_func=lambda x: nombres_map.get(x, x),
        label_visibility="collapsed",
    )

    cat_row  = data.catalogo[data.catalogo["id_producto"] == producto_sel].iloc[0]
    margen_u = cat_row["precio_venta"] - cat_row["costo_unitario"]
    tau_star = margen_u / (margen_u + cat_row["costo_almacenamiento_semanal"])

    st.markdown(f"""
    <hr style='border-color:{GOLD}; opacity:0.3; margin:18px 0 14px 0;'>
    <p style='color:{GOLD}; font-size:0.78rem; font-weight:700; letter-spacing:0.08em;'>FICHA DEL PRODUCTO</p>
    <table style='width:100%; font-size:0.82rem; color:#D5C5B0; border-collapse:collapse;'>
      <tr><td style='padding:3px 0;'>Categoria</td><td style='text-align:right; color:{CREAM};'>{cat_row['categoria']}</td></tr>
      <tr><td style='padding:3px 0;'>Costo unitario</td><td style='text-align:right; color:{CREAM};'>${cat_row['costo_unitario']:,}</td></tr>
      <tr><td style='padding:3px 0;'>Precio venta</td><td style='text-align:right; color:{CREAM};'>${cat_row['precio_venta']:,}</td></tr>
      <tr><td style='padding:3px 0;'>Margen</td><td style='text-align:right; color:{GOLD}; font-weight:700;'>${margen_u:,}</td></tr>
      <tr><td style='padding:3px 0;'>Almacenamiento/sem</td><td style='text-align:right; color:{CREAM};'>${cat_row['costo_almacenamiento_semanal']:,}</td></tr>
      <tr><td style='padding:3px 0;'>Fractil critico τ*</td><td style='text-align:right; color:{GOLD}; font-weight:700;'>{tau_star:.1%}</td></tr>
    </table>
    <hr style='border-color:{GOLD}; opacity:0.3; margin:14px 0;'>
    <p style='font-size:0.7rem; color:#9A8070; text-align:center;'>
      Modelo: LightGBM cuantil<br>Optimizacion: Newsvendor<br>Datos: 1 ene – 31 mar 2024
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS FILTRADOS
# ══════════════════════════════════════════════════════════════════════════════

nombre_prod = nombres_map.get(producto_sel, producto_sel)
res_prod    = results[results["id_producto"] == producto_sel].copy()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='padding:6px 0 20px 0;'>
  <h1 style='margin:0; font-size:1.65rem;'>Panel de Abastecimiento — {nombre_prod}</h1>
  <p style='color:#999; margin:4px 0 0 0; font-size:0.88rem;'>
    20 tiendas · 3 ciudades · Pronostico semana 1–7 abr 2024
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

dem_total   = res_prod["p50"].sum()
stock_total = res_prod["stock_actual"].sum()
pedido_tot  = res_prod["pedido_recomendado"].sum()
ahorro_tot  = res_prod["ahorro_estimado"].sum()
n_criticas  = (res_prod["stock_actual"] < res_prod["p10"] / 7 * 2).sum()

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Demanda esperada (sem · 20 tiendas)", f"{dem_total:.0f} uds",
                   f"p10:{res_prod['p10'].sum():.0f} – p90:{res_prod['p90'].sum():.0f}")
with c2: st.metric("Stock total actual", f"{stock_total:.0f} uds",
                   f"{stock_total/max(dem_total,1)*7:.1f} dias cobertura prom.")
with c3: st.metric("Pedido total recomendado", f"{int(pedido_tot)} uds",
                   f"en {(res_prod['pedido_recomendado']>0).sum()} de 20 tiendas")
with c4: st.metric("Ahorro estimado (sem)", f"${ahorro_tot/1e6:.1f}M COP",
                   f"anualizado: ${ahorro_tot*52/1e6:.0f}M COP")
with c5: st.metric("Tiendas en riesgo critico", f"{n_criticas} / 20",
                   "stock < 2 dias de cobertura")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Vision por tienda",
    "🌡️  Mapa de cobertura",
    "📈  Detalle de tienda",
    "💰  Impacto de negocio",
])


# ── TAB 1: Vision general ────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### Todas las tiendas — {nombre_prod}")
    st.markdown(
        "**Barra dorada** = demanda esperada (p50) con banda de incertidumbre (p10-p90). "
        "**Diamante rojo** = stock por debajo del escenario pesimista (p10): riesgo critico."
    )

    fig_ov, df_ov = chart_stores_overview(res_prod, data.ventas, producto_sel)
    st.pyplot(fig_ov, use_container_width=True)
    plt.close(fig_ov)

    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
    st.markdown("#### Tabla resumen por tienda")

    def riesgo_tag(dias):
        if dias < 2:  return "🔴 Critico"
        if dias < 5:  return "🟡 Bajo"
        return "🟢 OK"

    tabla = df_ov[[
        "id_tienda","ciudad","stock_actual","dem_diaria","dias_cob",
        "p10","p50","p90","pedido_recomendado","ahorro_estimado"
    ]].copy()
    tabla["riesgo"] = tabla["dias_cob"].apply(riesgo_tag)
    tabla = tabla.sort_values("dias_cob")

    st.dataframe(
        tabla.rename(columns={
            "id_tienda":"Tienda","ciudad":"Ciudad","stock_actual":"Stock",
            "dem_diaria":"Dem. diaria","dias_cob":"Dias cob.",
            "p10":"P10 sem","p50":"P50 sem","p90":"P90 sem",
            "pedido_recomendado":"Pedido rec.","ahorro_estimado":"Ahorro ($COP)",
            "riesgo":"Riesgo",
        }).round(1),
        use_container_width=True,
        height=380,
        column_config={
            "Ahorro ($COP)": st.column_config.NumberColumn(format="$%d"),
            "Stock": st.column_config.NumberColumn(format="%d"),
            "Pedido rec.": st.column_config.NumberColumn(format="%d"),
        }
    )


# ── TAB 2: Mapa de calor de cobertura ───────────────────────────────────────
with tab2:
    st.markdown("### Mapa de cobertura de inventario — todas las tiendas × todos los productos")
    st.markdown(
        "Muestra cuantos **dias de demanda** puede cubrir el inventario actual "
        "para cada combinacion tienda-producto. "
        "Identifica de un vistazo donde hay riesgo y donde hay exceso."
    )
    try:
        import seaborn as sns
        fig_hm = chart_heatmap_cobertura(data.ventas, data.inventario, data.tiendas, data.catalogo)
        st.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)
    except ImportError:
        st.warning("pip install seaborn para ver el mapa de calor")

    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
    st.markdown("#### Resumen por ciudad (todos los productos)")
    res_all = results.copy()
    res_ciudad = res_all.groupby("ciudad").agg(
        series        = ("id_tienda", "count"),
        stock_total   = ("stock_actual", "sum"),
        dem_p50_total = ("p50", "sum"),
        pedidos_total = ("pedido_recomendado", "sum"),
        ahorro_total  = ("ahorro_estimado", "sum"),
    ).round(0).reset_index()
    res_ciudad.columns = ["Ciudad","Series","Stock total","Dem. esperada (sem)","Pedidos rec.","Ahorro ($COP)"]
    st.dataframe(res_ciudad, use_container_width=True, hide_index=True,
                 column_config={"Ahorro ($COP)": st.column_config.NumberColumn(format="$%d")})


# ── TAB 3: Detalle de tienda ─────────────────────────────────────────────────
with tab3:
    st.markdown(f"### Detalle por tienda — {nombre_prod}")

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        tienda_sel = st.selectbox("Selecciona tienda",
                                  sorted(res_prod["id_tienda"].unique()))

    row = res_prod[res_prod["id_tienda"] == tienda_sel].iloc[0]
    ciudad_info = data.tiendas[data.tiendas["id_tienda"]==tienda_sel]["ciudad"].values[0]
    dem_diaria  = (data.ventas[
        (data.ventas["id_tienda"]==tienda_sel) &
        (data.ventas["id_producto"]==producto_sel)
    ]["unidades_vendidas"].mean())
    dias_cob = row["stock_actual"] / max(dem_diaria, 0.01)

    if dias_cob < 2:
        st.markdown(f'<div class="alert-red">Riesgo CRITICO — stock cubre solo {dias_cob:.1f} dias. '
                    f'Pedido urgente: <b>{int(row["pedido_recomendado"])} unidades</b>.</div>',
                    unsafe_allow_html=True)
    elif dias_cob < 5:
        st.markdown(f'<div class="alert-yellow">Riesgo BAJO — stock cubre {dias_cob:.1f} dias. '
                    f'Pedido recomendado: <b>{int(row["pedido_recomendado"])} unidades</b>.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-green">Stock OK — cobertura de {dias_cob:.1f} dias. '
                    f'Pedido recomendado: <b>{int(row["pedido_recomendado"])} unidades</b>.</div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Ciudad", ciudad_info)
    with k2: st.metric("Stock actual", f"{int(row['stock_actual'])} uds", f"{dias_cob:.1f} dias")
    with k3: st.metric("Pedido recomendado", f"{int(row['pedido_recomendado'])} uds")
    with k4: st.metric("Ahorro estimado", f"${row['ahorro_estimado']:,.0f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Forecast
    st.markdown("#### Historico de ventas + pronostico (unidades / dia)")
    st.caption("Banda dorada = intervalo de incertidumbre. Linea punteada = pronostico central diario (total semanal / 7).")
    fig_fc = chart_forecast_serie(data.ventas, tienda_sel, producto_sel, row, nombre_prod)
    st.pyplot(fig_fc, use_container_width=True)
    plt.close(fig_fc)

    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)

    # Newsvendor
    st.markdown("#### Decision de pedido — Modelo Newsvendor")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig_nv = chart_newsvendor(row)
        st.pyplot(fig_nv, use_container_width=True)
        plt.close(fig_nv)

    with col_b:
        ratio = int(margen_u / cat_row['costo_almacenamiento_semanal'])
        # FIX: usar info-box con HTML en vez de blockquote para mantener tipografia
        st.markdown(f"""
        | Parametro | Valor |
        |---|---|
        | Demanda p10 (sem) | {row['p10']:.0f} uds |
        | Demanda p50 (sem) | {row['p50']:.0f} uds |
        | Demanda p90 (sem) | {row['p90']:.0f} uds |
        | Stock actual | {int(row['stock_actual'])} uds |
        | Q* optimo | {row['cantidad_optima_pedido']:.0f} uds |
        | **Pedido recomendado** | **{int(row['pedido_recomendado'])} uds** |
        | Fractil critico τ* | {row['fractil_critico']:.1%} |
        """)
        st.markdown(
            f'<div class="info-box">El margen <b>${margen_u:,}</b> es <b>{ratio}x</b> '
            f'mayor que el costo de almacenamiento <b>${cat_row["costo_almacenamiento_semanal"]:,}</b>. '
            f'Por eso τ* = <b>{tau_star:.1%}</b> — casi siempre conviene ser agresivo en el pedido.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)

    # Costos
    st.markdown("#### Comparacion de costos: Naive vs. Optimo")
    col_c, col_d = st.columns([1, 2])
    with col_c:
        fig_cost = chart_cost_bars(row)
        st.pyplot(fig_cost, use_container_width=True)
        plt.close(fig_cost)
    with col_d:
        ahorro_pct = ((row["costo_esperado_naive"] - row["costo_esperado_optimo"])
                      / max(row["costo_esperado_naive"], 1) * 100)
        st.markdown(f"""
        | Politica | Costo esperado | Ahorro |
        |---|---|---|
        | Naive (pedir p50) | ${row['costo_esperado_naive']:,.0f} | — |
        | **Optima (Newsvendor)** | **${row['costo_esperado_optimo']:,.0f}** | **${row['ahorro_estimado']:,.0f} ({ahorro_pct:.1f}%)** |
        """)
        st.markdown(
            f'<div class="info-box">Anualizado en <b>{tienda_sel}</b> para <b>{nombre_prod}</b>: '
            f'<b>${row["ahorro_estimado"]*52:,.0f} COP/año</b> de ahorro estimado.</div>',
            unsafe_allow_html=True
        )

    with st.expander("Ver metricas de validacion del modelo (walk-forward CV)"):
        with st.spinner("Calculando..."):
            cv_r = run_cv(feat)
        for r in cv_r:
            st.markdown(
                f"**Fold {r.fold+1}** `{r.val_start.date()} → {r.val_end.date()}` "
                f"— MAE: `{r.mae:.2f}` | RMSE: `{r.rmse:.2f}` | WAPE: `{r.wape:.1%}`"
            )
        st.success(
            f"WAPE promedio: {np.mean([r.wape for r in cv_r]):.1%}  |  "
            f"MAE promedio: {np.mean([r.mae for r in cv_r]):.2f} uds/dia"
        )


# ── TAB 4: Impacto de negocio ────────────────────────────────────────────────
with tab4:
    st.markdown("### Impacto economico — todos los productos · 20 tiendas")

    res_all = results.copy()
    ahorro_g = res_all["ahorro_estimado"].sum()
    pedido_g = res_all["pedido_recomendado"].sum()

    g1, g2, g3 = st.columns(3)
    with g1: st.metric("Ahorro total semanal", f"${ahorro_g/1e6:.1f}M COP")
    with g2: st.metric("Ahorro total anualizado", f"${ahorro_g*52/1e6:.0f}M COP")
    with g3: st.metric("Pedidos totales recomendados", f"{int(pedido_g):,} uds")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    por_prod = (res_all.groupby("nombre")["ahorro_estimado"]
                .sum().sort_values(ascending=True).reset_index())
    fig_p, ax_p = mpl_fig((9, 4))
    bars = ax_p.barh(por_prod["nombre"], por_prod["ahorro_estimado"],
                     color=GOLD, alpha=0.85, edgecolor=WHITE)
    for bar, val in zip(bars, por_prod["ahorro_estimado"]):
        ax_p.text(val+200, bar.get_y()+bar.get_height()/2,
                  f"${val/1e6:.1f}M", va="center", fontsize=9,
                  color=BROWN, fontweight="bold")
    ax_p.set_xlabel("Ahorro semanal estimado ($COP)")
    ax_p.set_title("Ahorro semanal por producto (suma 20 tiendas)\nPolitica Optima vs. Naive",
                   color=BROWN, fontweight="bold")
    ax_p.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: f"${v/1e6:.1f}M"))
    fig_p.tight_layout()
    st.pyplot(fig_p, use_container_width=True)
    plt.close(fig_p)

    st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
    st.markdown("#### Tabla completa de recomendaciones (160 series)")

    tabla_all = res_all[[
        "id_tienda","ciudad","nombre","stock_actual",
        "p50","pedido_recomendado","ahorro_estimado",
        "costo_esperado_naive","costo_esperado_optimo"
    ]].copy().sort_values("ahorro_estimado", ascending=False)
    for col in ["p50","stock_actual","pedido_recomendado","ahorro_estimado",
                "costo_esperado_naive","costo_esperado_optimo"]:
        tabla_all[col] = tabla_all[col].round(0)

    st.dataframe(
        tabla_all.rename(columns={
            "id_tienda":"Tienda","ciudad":"Ciudad","nombre":"Producto",
            "stock_actual":"Stock","p50":"Dem. esperada (sem)",
            "pedido_recomendado":"Pedido rec.","ahorro_estimado":"Ahorro ($COP)",
            "costo_esperado_naive":"Costo naive","costo_esperado_optimo":"Costo optimo",
        }),
        use_container_width=True,
        height=420,
        column_config={
            "Ahorro ($COP)": st.column_config.NumberColumn(format="$%d"),
            "Stock": st.column_config.NumberColumn(format="%d"),
            "Pedido rec.": st.column_config.NumberColumn(format="%d"),
        }
    )
