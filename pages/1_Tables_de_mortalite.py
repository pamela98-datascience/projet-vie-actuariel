"""
Page 1 - Tables de mortalité
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)
"""

import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.module1_mortalite import (
    get_table_mortalite,
    construire_table_complete,
    prob_survie_n_ans,
    prob_deces_n_ans,
    loi_de_survie,
    comparer_tables,
    exporter_table_csv,
)

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tables de mortalité", page_icon="📊", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1e293b; margin: 4px 0; }
    .metric-sub   { font-size: 12px; color: #94a3b8; }
    .section-tag  { display:inline-block; background:#dbeafe; color:#1e40af; font-size:11px;
                    font-weight:600; padding:2px 10px; border-radius:20px; margin-bottom:8px; }
    .formula-box  { background:#f0fdf4; border-left:3px solid #22c55e; padding:10px 16px;
                    border-radius:0 8px 8px 0; font-family:monospace; font-size:13px; color:#166534; }
</style>
""", unsafe_allow_html=True)

# ── En-tête ──────────────────────────────────────────────────────────────────
st.markdown('<span class="section-tag">MODULE 1 · LEÇONS 1–3</span>', unsafe_allow_html=True)
st.title("Tables de mortalité")
st.caption("Construction des indicateurs actuariels fondamentaux : qx · lx · dx · ex · loi de survie S(t)")

st.divider()

# ── Sidebar : paramètres ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")

    table_choisie = st.selectbox(
        "Table de mortalité",
        ["TH00-02", "TF00-02", "TD88-90"],
        help="TH = Hommes, TF = Femmes, TD = Toutes Décédants (ancienne)"
    )

    age_min, age_max = st.slider("Tranche d'âge analysée", 0, 110, (0, 100))

    age_ref = st.number_input("Âge de référence (calculs ponctuels)", min_value=0, max_value=100, value=40)

    horizon_n = st.number_input("Horizon n (probabilités n-ans)", min_value=1, max_value=50, value=10)

    l0 = st.number_input("Radix l₀ (effectif initial)", min_value=1000, max_value=1_000_000,
                          value=100_000, step=10_000)

    st.divider()
    st.markdown("**Rappel cours (leçon 2)**")
    st.info("qx = probabilité de décéder entre x et x+1\n\nlx = effectif survivant à x\n\ndx = lx × qx")

# ── Chargement des données ────────────────────────────────────────────────────
df_qx = get_table_mortalite(table_choisie)
df = construire_table_complete(df_qx, l0=l0)
df_filtre = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()

# ── Section 1 : KPIs ─────────────────────────────────────────────────────────
st.markdown("### Indicateurs clés")

ex0  = round(df["ex"].iloc[0], 2)
ex60 = round(df.loc[df["age"] == 60, "ex"].values[0], 2)
lx_ref = int(df.loc[df["age"] == age_ref, "lx"].values[0])
qx_ref = df.loc[df["age"] == age_ref, "qx"].values[0]
npx   = round(prob_survie_n_ans(df, age_ref, horizon_n) * 100, 2)
nqx   = round(prob_deces_n_ans(df, age_ref, horizon_n) * 100, 2)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Espérance de vie à 0</div>
        <div class="metric-value">{ex0} ans</div>
        <div class="metric-sub">{table_choisie}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Espérance résiduelle à 60</div>
        <div class="metric-value">{ex60} ans</div>
        <div class="metric-sub">retraite collective</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Survivants à {age_ref} ans</div>
        <div class="metric-value">{lx_ref:,}</div>
        <div class="metric-sub">sur {l0:,} initiaux</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">qx à {age_ref} ans</div>
        <div class="metric-value">{qx_ref*1000:.3f} ‰</div>
        <div class="metric-sub">probabilité annuelle décès</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">{horizon_n}p{age_ref} — survie {horizon_n} ans</div>
        <div class="metric-value">{npx} %</div>
        <div class="metric-sub">{horizon_n}q{age_ref} = {nqx} %</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2 : Graphiques ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 qx (log)", "👥 lx — effectifs", "⏳ Loi de survie S(t)", "📊 Comparaison tables"])

with tab1:
    st.markdown("#### Taux de mortalité qx par âge (échelle log)")
    st.markdown("""<div class="formula-box">qx = probabilité de décéder entre x et x+1<br>
    Lecture : une valeur élevée = fort risque de mortalité à cet âge</div>""", unsafe_allow_html=True)
    st.markdown("")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtre["age"],
        y=df_filtre["qx"] * 1000,
        mode="lines",
        name="qx (‰)",
        line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="Âge %{x}<br>qx = %{y:.4f} ‰<extra></extra>"
    ))
    fig.add_vline(x=age_ref, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"Âge {age_ref}", annotation_position="top")
    fig.update_layout(
        yaxis_type="log",
        yaxis_title="qx (pour mille) — échelle logarithmique",
        xaxis_title="Âge",
        height=400,
        margin=dict(t=20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Effectifs survivants lx")
    st.markdown("""<div class="formula-box">lx = l₀ × ₓp₀ = effectif survivant à l'âge x<br>
    dx = lx × qx = nombre de décès entre x et x+1</div>""", unsafe_allow_html=True)
    st.markdown("")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_filtre["age"], y=df_filtre["lx"],
        mode="lines", name="lx (survivants)",
        line=dict(color="#10b981", width=2.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="Âge %{x}<br>lx = %{y:,.0f}<extra></extra>"
    ))
    fig2.add_trace(go.Bar(
        x=df_filtre["age"], y=df_filtre["dx"],
        name="dx (décès)", marker_color="rgba(239,68,68,0.4)",
        yaxis="y2",
        hovertemplate="Âge %{x}<br>dx = %{y:,.1f}<extra></extra>"
    ))
    fig2.update_layout(
        height=400,
        yaxis=dict(title="Survivants lx", gridcolor="#f1f5f9"),
        yaxis2=dict(title="Décès dx", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="Âge", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("#### Loi de survie S(t) depuis un âge de départ")
    st.markdown("""<div class="formula-box">S(t) = P(T > t) = lx+t / lx = probabilité d'être en vie t ans après l'âge de départ<br>
    Fondamentale pour le calcul des provisions mathématiques et du Best Estimate vie</div>""", unsafe_allow_html=True)
    st.markdown("")

    age_depart = st.slider("Âge de départ pour S(t)", 0, 80, 35)
    s = loi_de_survie(df, age_depart)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(s.index), y=s.values * 100,
        mode="lines", name="S(t) (%)",
        line=dict(color="#8b5cf6", width=2.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
        hovertemplate="Âge %{x}<br>S(t) = %{y:.2f} %<extra></extra>"
    ))
    # Médiane de survie
    mediane = s[s <= 0.5]
    if len(mediane) > 0:
        age_med = mediane.index[0]
        fig3.add_vline(x=age_med, line_dash="dot", line_color="#f59e0b",
                       annotation_text=f"Médiane : {age_med} ans",
                       annotation_position="top right")
    fig3.update_layout(
        height=400,
        yaxis=dict(title="S(t) en %", gridcolor="#f1f5f9"),
        xaxis=dict(title="Âge", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

    ex_depart = df.loc[df["age"] == age_depart, "ex"].values[0]
    st.info(f"Espérance de vie résiduelle à {age_depart} ans : **{ex_depart:.2f} ans** — soit jusqu'à {age_depart + ex_depart:.0f} ans en moyenne")

with tab4:
    st.markdown("#### Comparaison TH00-02 · TF00-02 · TD88-90")
    st.markdown("""<div class="formula-box">Les tables évoluent dans le temps (amélioration de la mortalité).
    TD 88-90 > TH 00-02 > TF 00-02 en termes de mortalité.</div>""", unsafe_allow_html=True)
    st.markdown("")

    ages_comp = list(range(0, 101, 5))
    fig4 = go.Figure()
    colors = {"TH00-02": "#3b82f6", "TF00-02": "#ec4899", "TD88-90": "#6b7280"}

    for t in ["TH00-02", "TF00-02", "TD88-90"]:
        df_t = construire_table_complete(get_table_mortalite(t), l0=100_000)
        df_t_f = df_t[df_t["age"].isin(ages_comp)]
        fig4.add_trace(go.Scatter(
            x=df_t_f["age"], y=df_t_f["qx"] * 1000,
            mode="lines+markers", name=t,
            line=dict(color=colors[t], width=2),
            marker=dict(size=5),
            hovertemplate=f"{t}<br>Âge %{{x}}<br>qx = %{{y:.4f}} ‰<extra></extra>"
        ))

    fig4.update_layout(
        yaxis_type="log",
        height=420,
        yaxis=dict(title="qx (‰) — log", gridcolor="#f1f5f9"),
        xaxis=dict(title="Âge", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Tableau comparatif — espérance de vie résiduelle ex**")
    df_comp = comparer_tables([0, 20, 30, 40, 50, 60, 65, 70, 75, 80])
    df_pivot = df_comp.pivot(index="Age", columns="Table", values="ex (années)").reset_index()
    st.dataframe(df_pivot, use_container_width=True, hide_index=True)

# ── Section 3 : Table complète ────────────────────────────────────────────────
st.divider()
st.markdown("### Table actuarielle complète")

cols_affich = ["age", "qx", "px", "lx", "dx", "ex"]
df_display = df_filtre[cols_affich].copy()
df_display["qx (‰)"] = (df_display["qx"] * 1000).round(4)
df_display["lx"] = df_display["lx"].round(0).astype(int)
df_display["dx"] = df_display["dx"].round(2)
df_display["ex"] = df_display["ex"].round(2)
df_display = df_display.rename(columns={"px": "px", "qx": "qx (décimal)"})

st.dataframe(
    df_display[["age", "qx (‰)", "px", "lx", "dx", "ex"]],
    use_container_width=True, hide_index=True,
    column_config={
        "age": st.column_config.NumberColumn("Âge", format="%d"),
        "qx (‰)": st.column_config.NumberColumn("qx (‰)", format="%.4f"),
        "px": st.column_config.NumberColumn("px", format="%.6f"),
        "lx": st.column_config.NumberColumn("lx", format="%d"),
        "dx": st.column_config.NumberColumn("dx", format="%.2f"),
        "ex": st.column_config.NumberColumn("ex (ans)", format="%.2f"),
    }
)

# ── Section 4 : Formules cours ────────────────────────────────────────────────
st.divider()
with st.expander("Rappel des formules — cours ISUP Leçons 1–3"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Loi de mortalité de base**")
        st.latex(r"q_x = P(T \leq x+1 \mid T > x)")
        st.latex(r"p_x = 1 - q_x")
        st.latex(r"l_{x+1} = l_x \cdot p_x")
        st.latex(r"d_x = l_x \cdot q_x")
    with col2:
        st.markdown("**Probabilités n-ans**")
        st.latex(r"_np_x = \frac{l_{x+n}}{l_x}")
        st.latex(r"_nq_x = 1 - {_np_x}")
        st.latex(r"e_x = \frac{T_x}{l_x} = \sum_{k=0}^{\omega - x} {_kp_x}")
        st.latex(r"S(t) = {_tp_0} = \frac{l_t}{l_0}")

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Export des données")
col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    csv = df_filtre.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button(
        "Télécharger la table (CSV pour SAS)",
        data=csv,
        file_name=f"table_{table_choisie}_ages{age_min}_{age_max}.csv",
        mime="text/csv",
        use_container_width=True
    )
with col_exp2:
    st.info("Ce CSV est formaté avec séparateur `;` et virgule décimale pour import direct en SAS (PROC IMPORT).", icon="ℹ️")

