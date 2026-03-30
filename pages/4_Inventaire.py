"""
Page 4 - Inventaire & Résultats
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)
"""

import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.module1_mortalite import get_table_mortalite, construire_table_complete
from modules.module3_PM import PM_prospective
from modules.module4_inventaire import (
    inventaire_annuel, simulation_multi_annees,
    duration_PM, tableau_duration, pb_detail
)

st.set_page_config(page_title="Inventaire & Résultats", page_icon="📋", layout="wide")

st.markdown("""
<style>
    .metric-card { background:#f8f9fc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 18px; text-align:center; }
    .metric-label { font-size:11px; color:#64748b; font-weight:500;
        text-transform:uppercase; letter-spacing:.05em; }
    .metric-value { font-size:24px; font-weight:700; color:#1e293b; margin:4px 0; }
    .metric-sub   { font-size:11px; color:#94a3b8; }
    .section-tag  { display:inline-block; background:#fef9c3; color:#713f12;
        font-size:11px; font-weight:600; padding:2px 10px;
        border-radius:20px; margin-bottom:8px; }
    .formula-box  { background:#fffbeb; border-left:3px solid #f59e0b;
        padding:10px 16px; border-radius:0 8px 8px 0;
        font-family:monospace; font-size:13px; color:#78350f; }
    .pos { color: #16a34a; font-weight:600; }
    .neg { color: #dc2626; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="section-tag">MODULE 4 · LEÇONS 8–10</span>', unsafe_allow_html=True)
st.title("Inventaire & Résultats Techniques")
st.caption("Compte technique · Résultat financier / mortalité / gestion · PB · Duration ALM")
st.divider()

# ── Chargement ──────────────────────────────────────────────────────────────
df_th = construire_table_complete(get_table_mortalite("TH00-02"))

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Portefeuille")
    age_0       = st.number_input("Âge moyen départ", 60, 75, 65)
    rente_R     = st.number_input("Rente annuelle/assuré (€)", 1000, 30000, 14400, 1000)
    n_assures   = st.number_input("Nombre d'assurés", 100, 5000, 1000, 100)
    taux_i      = st.slider("Taux technique i (%)", 0.0, 5.0, 2.0, 0.1) / 100
    taux_gest   = st.slider("Chargement gestion β (%)", 0.0, 2.0, 0.8, 0.1) / 100
    taux_pen    = st.slider("Pénalité rachat (%)", 0.0, 5.0, 2.0, 0.5) / 100

    st.divider()
    st.markdown("**Scénarios annuels**")
    st.caption("Année 1")
    rend1  = st.slider("Rendement A1 (%)", 0.0, 8.0, 3.5, 0.1) / 100
    mort1  = st.slider("Mortalité A1 (% att.)", 50, 150, 85, 5) / 100
    rach1  = st.slider("Rachat A1 (%)", 0.0, 10.0, 2.0, 0.5) / 100
    st.caption("Année 2")
    rend2  = st.slider("Rendement A2 (%)", 0.0, 8.0, 2.2, 0.1) / 100
    mort2  = st.slider("Mortalité A2 (% att.)", 50, 150, 100, 5) / 100
    rach2  = st.slider("Rachat A2 (%)", 0.0, 10.0, 3.0, 0.5) / 100
    st.caption("Année 3")
    rend3  = st.slider("Rendement A3 (%)", 0.0, 8.0, 1.2, 0.1) / 100
    mort3  = st.slider("Mortalité A3 (% att.)", 50, 150, 75, 5) / 100
    rach3  = st.slider("Rachat A3 (%)", 0.0, 10.0, 4.0, 0.5) / 100

# ── Scénarios ──────────────────────────────────────────────────────────────
scenarios = [
    {"annee": 1, "rendement_reel": rend1, "tx_mortalite_reel": mort1,
     "frais_reels_pct": taux_gest * 0.75, "tx_rachat_reel": rach1},
    {"annee": 2, "rendement_reel": rend2, "tx_mortalite_reel": mort2,
     "frais_reels_pct": taux_gest, "tx_rachat_reel": rach2},
    {"annee": 3, "rendement_reel": rend3, "tx_mortalite_reel": mort3,
     "frais_reels_pct": taux_gest * 1.1, "tx_rachat_reel": rach3},
]

df_inv = simulation_multi_annees(
    df_th, PM_prospective, age_0, rente_R, taux_i,
    n_assures, taux_gest, taux_pen, scenarios
)

PM_0_total = PM_prospective(df_th, age_0, rente_R, taux_i) * n_assures

# ── KPIs globaux ────────────────────────────────────────────────────────────
st.markdown("### Indicateurs portefeuille")
res_total_3ans = df_inv["Rés. total (€)"].sum()
PB_total_3ans  = df_inv["PB totale (€)"].sum()
rentes_total   = df_inv["Rentes versées (€)"].sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PM initiale</div>
        <div class="metric-value">{PM_0_total/1e6:.1f} M€</div>
        <div class="metric-sub">{n_assures:,} assurés</div>
    </div>""", unsafe_allow_html=True)
with c2:
    signe = "+" if res_total_3ans >= 0 else ""
    couleur = "#16a34a" if res_total_3ans >= 0 else "#dc2626"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Résultat 3 ans</div>
        <div class="metric-value" style="color:{couleur}">{signe}{res_total_3ans/1e6:.2f} M€</div>
        <div class="metric-sub">cumulé</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PB totale 3 ans</div>
        <div class="metric-value">{PB_total_3ans/1e6:.2f} M€</div>
        <div class="metric-sub">redistribuée aux assurés</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Rentes versées 3 ans</div>
        <div class="metric-value">{rentes_total/1e6:.1f} M€</div>
        <div class="metric-sub">flux sortants</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Onglets ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Inventaire détaillé", "🔍 Décomposition résultat",
    "💰 Participation aux bénéfices", "📐 Duration & ALM"
])

with tab1:
    st.markdown("#### Inventaire année par année")
    st.markdown("""<div class="formula-box">
    Équation de passage : PM_fin = PM_deb + Intérêts − Rentes − Rachats ± Δ mortalité + PB
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    for _, row in df_inv.iterrows():
        annee = int(row["Année"])
        res   = row["Rés. total (€)"]
        icone = "🟢" if res >= 0 else "🔴"
        with st.expander(f"{icone} Année {annee} — Résultat : {res:+,.0f} €  |  Rendement {row['Rendement réel']}  |  Mortalité {row['Mortalité (% att.)']}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Assurés début", f"{int(row['Assurés début']):,}")
                st.metric("Décès attendus", f"{row['Décès attendus']:.1f}")
                st.metric("Décès réels", f"{row['Décès réels']:.1f}")
            with c2:
                st.metric("Rachats", f"{row['Rachats']:.1f}")
                st.metric("Assurés fin", f"{int(row['Assurés fin']):,}")
                st.metric("Rentes versées", f"{row['Rentes versées (€)']:,.0f} €")
            with c3:
                st.metric("PM début", f"{row['PM début (€)']:,.0f} €")
                st.metric("PM fin", f"{row['PM fin (€)']:,.0f} €")
                st.metric("PB versée", f"{row['PB totale (€)']:,.0f} €")

    st.markdown("**Tableau récapitulatif**")
    cols_affich = ["Année", "Assurés début", "Assurés fin",
                   "PM début (€)", "PM fin (€)", "Rés. total (€)", "PB totale (€)"]
    st.dataframe(
        df_inv[cols_affich], use_container_width=True, hide_index=True,
        column_config={
            "PM début (€)":   st.column_config.NumberColumn(format="%,.0f"),
            "PM fin (€)":     st.column_config.NumberColumn(format="%,.0f"),
            "Rés. total (€)": st.column_config.NumberColumn(format="%+,.0f"),
            "PB totale (€)":  st.column_config.NumberColumn(format="%,.0f"),
        }
    )

with tab2:
    st.markdown("#### Décomposition du résultat technique")
    st.markdown("""<div class="formula-box">
    Résultat = Résultat financier + Résultat mortalité + Résultat gestion<br>
    Résultat financier  = (rendement réel − i) × PM<br>
    Résultat mortalité  = −(décès att. − décès réels) × PM/tête  [négatif si longévité]<br>
    Résultat gestion    = chargements perçus − frais réels
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    annees = df_inv["Année"].values
    x = np.arange(len(annees))
    w = 0.25

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x - w, y=df_inv["Rés. financier (€)"], width=w,
                         name="Financier", marker_color="#3b82f6",
                         hovertemplate="A%{x}<br>%{y:+,.0f}€<extra>Financier</extra>"))
    fig.add_trace(go.Bar(x=x,     y=df_inv["Rés. mortalité (€)"], width=w,
                         name="Mortalité", marker_color="#ef4444",
                         hovertemplate="A%{x}<br>%{y:+,.0f}€<extra>Mortalité</extra>"))
    fig.add_trace(go.Bar(x=x + w, y=df_inv["Rés. gestion (€)"],  width=w,
                         name="Gestion", marker_color="#8b5cf6",
                         hovertemplate="A%{x}<br>%{y:+,.0f}€<extra>Gestion</extra>"))
    fig.add_trace(go.Scatter(x=x, y=df_inv["Rés. total (€)"],
                             mode="lines+markers", name="Total",
                             line=dict(color="black", width=2.5),
                             marker=dict(size=9, symbol="diamond"),
                             hovertemplate="Total : %{y:+,.0f}€<extra></extra>"))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        height=420,
        xaxis=dict(tickvals=list(x),
                   ticktext=[f"Année {a}" for a in annees],
                   gridcolor="#f1f5f9"),
        yaxis=dict(title="Résultat (€)", tickformat="+,.0f", gridcolor="#f1f5f9"),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Interprétation automatique**")
    for _, row in df_inv.iterrows():
        a   = int(row["Année"])
        rf  = row["Rés. financier (€)"]
        rm  = row["Rés. mortalité (€)"]
        rg  = row["Rés. gestion (€)"]
        tot = row["Rés. total (€)"]
        dominant = max([(abs(rf), "financier"), (abs(rm), "mortalité"),
                        (abs(rg), "gestion")], key=lambda x: x[0])[1]
        signe_tot = "bénéficiaire" if tot >= 0 else "déficitaire"
        st.markdown(
            f"**Année {a}** — Résultat **{signe_tot}** ({tot:+,.0f}€). "
            f"Effet dominant : **{dominant}** ({rf if dominant=='financier' else rm if dominant=='mortalité' else rg:+,.0f}€)."
        )

with tab3:
    st.markdown("#### Participation aux bénéfices (PB)")
    st.markdown("""<div class="formula-box">
    PB légale minimum : 85% × Résultat financier + 90% × Résultat technique<br>
    PPB : provision pour PB — réserve de lissage pour les années déficitaires
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=[f"Année {a}" for a in df_inv["Année"]],
        y=df_inv["PB totale (€)"],
        marker_color="#10b981", opacity=0.8,
        hovertemplate="%{x}<br>PB = %{y:,.0f} €<extra></extra>"
    ))
    fig3.update_layout(
        height=340,
        yaxis=dict(title="PB (€)", tickformat=",.0f", gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Détail de la PB par année**")
    for _, row in df_inv.iterrows():
        a = int(row["Année"])
        rf = row["Rés. financier (€)"]
        rm = row["Rés. mortalité (€)"]
        rg = row["Rés. gestion (€)"]
        n  = int(row["Assurés début"])
        pb = pb_detail(rf, rm, rg, n)
        with st.expander(f"Année {a} — PB = {pb['PB TOTALE (€)']:,.0f} €"):
            for k, v in pb.items():
                if "€" in k:
                    signe = "+" if v >= 0 else ""
                    st.markdown(f"**{k}** : {signe}{v:,.0f} €")
                else:
                    st.markdown(f"**{k}** : {v}")

    st.info("En année 3 (résultat négatif), aucune PB légale n'est due. "
            "L'assureur peut puiser dans sa PPB accumulée pour lisser "
            "l'impact sur les assurés.")

with tab4:
    st.markdown("#### Duration de Macaulay et gestion actif-passif (ALM)")
    st.markdown("""<div class="formula-box">
    Duration = Σ(t × flux_t) / Σ(flux_t)<br>
    Mesure la sensibilité de la PM aux taux. L'actif doit avoir une duration similaire (immunisation).
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    ages_dur = list(range(60, 86, 5))
    df_dur   = tableau_duration(df_th, ages_dur, rente_R, taux_i)
    dur_ref  = duration_PM(df_th, age_0, rente_R, taux_i)
    dur_mod  = dur_ref / (1 + taux_i)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Duration PM à {age_0} ans", f"{dur_ref:.2f} ans")
    with c2:
        st.metric("Duration modifiée", f"{dur_mod:.2f}")
    with c3:
        st.metric("Sensibilité −1% taux", f"+{dur_mod:.2f}% PM",
                  delta=f"+{PM_0_total * dur_mod / 100:,.0f} €",
                  delta_color="inverse")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_dur["Âge"], y=df_dur["Duration (ans)"],
        mode="lines+markers", name="Duration",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=7),
        hovertemplate="Âge %{x}<br>Duration = %{y:.2f} ans<extra></extra>"
    ))
    fig4.add_vline(x=age_0, line_dash="dot", line_color="#8b5cf6",
                   annotation_text=f"Âge départ {age_0} ans")
    fig4.update_layout(
        height=340,
        xaxis=dict(title="Âge de l'assuré", gridcolor="#f1f5f9"),
        yaxis=dict(title="Duration (années)", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(df_dur, use_container_width=True, hide_index=True)
    st.markdown(f"""
    **Implication ALM :** l'actif doit avoir une duration ≈ **{dur_ref:.1f} ans**
    pour immuniser le portefeuille contre le risque de taux.
    On privilégie des **obligations longues** (OAT 20-30 ans) pour matcher
    la duration du passif retraite.
    """)

# ── Formules ────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Rappel des formules — cours ISUP Leçons 8–10"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Équation de l'inventaire**")
        st.latex(r"PM_{fin} = PM_{deb} + Prod_{fin} - Rentes - Rachats + PB")
        st.markdown("**Résultat financier**")
        st.latex(r"RF = (r_{réel} - i) \times PM_{deb}")
        st.markdown("**Résultat mortalité**")
        st.latex(r"RM = -(d_{att} - d_{réels}) \times \frac{PM}{n}")
    with col2:
        st.markdown("**Participation aux bénéfices**")
        st.latex(r"PB \geq 0.85 \times RF + 0.90 \times RT")
        st.markdown("**Duration de Macaulay**")
        st.latex(r"D = \frac{\sum_t t \cdot flux_t}{\sum_t flux_t}")
        st.markdown("**Duration modifiée**")
        st.latex(r"D^* = \frac{D}{1+i} \Rightarrow \frac{\Delta PM}{PM} \approx -D^* \cdot \Delta i")
