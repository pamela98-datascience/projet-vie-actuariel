"""
Page 5 - Backtesting PM (Analyse N/N+1)
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)
"""

import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from modules.module1_mortalite import get_table_mortalite, construire_table_complete
from modules.module3_PM import PM_prospective
from modules.module5_backtesting import (
    construire_portefeuille, calculer_PM_portefeuille,
    simuler_evenements, backtesting_N_N1,
    decomposer_ecart, analyse_par_age, rapport_backtesting
)

st.set_page_config(page_title="Backtesting PM", page_icon="🔁", layout="wide")

st.markdown("""
<style>
    .metric-card { background:#f8f9fc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 18px; text-align:center; }
    .metric-label { font-size:11px; color:#64748b; font-weight:500;
        text-transform:uppercase; letter-spacing:.05em; }
    .metric-value { font-size:24px; font-weight:700; color:#1e293b; margin:4px 0; }
    .metric-sub   { font-size:11px; color:#94a3b8; }
    .section-tag  { display:inline-block; background:#fee2e2; color:#991b1b;
        font-size:11px; font-weight:600; padding:2px 10px;
        border-radius:20px; margin-bottom:8px; }
    .formula-box  { background:#fff1f2; border-left:3px solid #ef4444;
        padding:10px 16px; border-radius:0 8px 8px 0;
        font-family:monospace; font-size:13px; color:#7f1d1d; }
    .rapport-box  { background:#f0fdf4; border:1px solid #bbf7d0;
        border-radius:8px; padding:16px; font-family:monospace;
        font-size:12px; color:#14532d; white-space:pre; }
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="section-tag">MODULE 5 · ANALYSE N/N+1</span>', unsafe_allow_html=True)
st.title("Backtesting PM — Analyse N/N+1")
st.caption("PM attendue vs réelle · Décomposition écart · Révision des hypothèses")
st.divider()

# ── Chargement ──────────────────────────────────────────────────────────────
df_th = construire_table_complete(get_table_mortalite("TH00-02"))

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres backtesting")
    n_contrats = st.number_input("Nombre de contrats", 50, 500, 200, 50)
    taux_N     = st.slider("Taux technique à N (%)", 0.5, 5.0, 2.0, 0.1) / 100
    taux_N1    = st.slider("Taux technique à N+1 (%)", 0.5, 5.0, 1.5, 0.1) / 100
    fact_mort  = st.slider("Mortalité réelle (% table)", 50, 130, 80, 5) / 100
    tx_rachat  = st.slider("Taux rachat réel (%)", 0.0, 10.0, 4.0, 0.5) / 100
    tx_rack_att = st.slider("Taux rachat attendu (%)", 0.0, 10.0, 3.0, 0.5) / 100
    seed       = st.number_input("Graine aléatoire", 1, 999, 42)
    st.divider()
    st.info("Écart > 0 → sous-provisionnement\nÉcart < 0 → sur-provisionnement")

# ── Construction portefeuille ───────────────────────────────────────────────
port_base = construire_portefeuille(n_contrats, seed)
port_pm   = calculer_PM_portefeuille(port_base, df_th, PM_prospective, taux_N)
port_evt  = simuler_evenements(port_pm, df_th, fact_mort, tx_rachat, seed)
port_bt   = backtesting_N_N1(port_evt, df_th, PM_prospective, taux_N, taux_N1)

n_deces   = port_evt["deces"].sum()
n_rachats = port_evt["rachat"].sum()
n_survie  = port_evt["survie"].sum()

total_att  = port_bt["PM_N1_attendue"].sum()
total_reel = port_bt["PM_N1_reelle"].sum()
ecart_tot  = total_reel - total_att

decomp = decomposer_ecart(
    port_evt, df_th, PM_prospective,
    taux_N, taux_N1, fact_mort,
    n_deces, n_rachats, n_contrats * tx_rack_att
)

# ── KPIs ────────────────────────────────────────────────────────────────────
st.markdown("### Résultats du backtesting")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Contrats survivants</div>
        <div class="metric-value">{n_survie}</div>
        <div class="metric-sub">sur {n_contrats} initiaux</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PM attendue N+1</div>
        <div class="metric-value">{total_att/1e6:.2f} M€</div>
        <div class="metric-sub">hypothèses taux {taux_N*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PM réelle N+1</div>
        <div class="metric-value">{total_reel/1e6:.2f} M€</div>
        <div class="metric-sub">taux réel {taux_N1*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c4:
    couleur = "#dc2626" if ecart_tot > 0 else "#16a34a"
    label   = "SOUS-PROV." if ecart_tot > 0 else "SUR-PROV."
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Écart total ({label})</div>
        <div class="metric-value" style="color:{couleur}">{ecart_tot/1e6:+.2f} M€</div>
        <div class="metric-sub">{ecart_tot/total_att*100:+.2f}% de la PM</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Onglets ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portefeuille & événements", "🔍 Analyse des écarts",
    "📉 Décomposition", "📄 Rapport"
])

with tab1:
    st.markdown("#### Portefeuille à date N")
    st.markdown("""<div class="formula-box">
    Portefeuille hétérogène : âges 62-68 ans, rentes 9 600-24 000 €/an.
    Entre N et N+1 : décès (selon table × facteur), rachats, survivants.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(port_pm, x="age_N", nbins=7,
                               title="Distribution des âges",
                               color_discrete_sequence=["#3b82f6"])
        fig_age.update_layout(height=300, plot_bgcolor="white",
                               paper_bgcolor="white",
                               xaxis_title="Âge", yaxis_title="Nb contrats",
                               margin=dict(t=40, b=30))
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        fig_rente = px.histogram(port_pm, x="rente", nbins=5,
                                 title="Distribution des rentes",
                                 color_discrete_sequence=["#8b5cf6"])
        fig_rente.update_layout(height=300, plot_bgcolor="white",
                                 paper_bgcolor="white",
                                 xaxis_title="Rente (€/an)",
                                 yaxis_title="Nb contrats",
                                 margin=dict(t=40, b=30))
        st.plotly_chart(fig_rente, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Décès", f"{n_deces}",
                  delta=f"{n_deces/n_contrats*100:.1f}%")
    with col4:
        st.metric("Rachats", f"{n_rachats}",
                  delta=f"{n_rachats/n_contrats*100:.1f}%")
    with col5:
        st.metric("Survivants", f"{n_survie}",
                  delta=f"{n_survie/n_contrats*100:.1f}%")

    qx_att_total = port_pm.apply(
        lambda r: df_th.loc[df_th["age"]==int(r["age_N"]), "qx"].values[0], axis=1
    ).sum()
    st.info(f"Décès attendus selon table : **{qx_att_total:.1f}** — "
            f"Décès réels : **{n_deces}** "
            f"(mortalité à {fact_mort*100:.0f}% de la table TH 00-02)")

with tab2:
    st.markdown("#### PM attendue vs PM réelle par contrat")
    st.markdown("""<div class="formula-box">
    PM attendue = recalcul à age+1 avec hypothèses N (taux inchangé)<br>
    PM réelle   = recalcul à age+1 avec taux N+1 (marché)<br>
    Tout point au-dessus de la diagonale = sous-provisionnement
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=port_bt["PM_N1_attendue"] / 1000,
            y=port_bt["PM_N1_reelle"] / 1000,
            mode="markers",
            marker=dict(color="#3b82f6", size=5, opacity=0.5),
            hovertemplate="Att: %{x:.0f}k€<br>Réel: %{y:.0f}k€<extra></extra>"
        ))
        lim = max(port_bt["PM_N1_attendue"].max(),
                  port_bt["PM_N1_reelle"].max()) / 1000 * 1.05
        fig_scatter.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="Attendu = Réel"
        ))
        fig_scatter.update_layout(
            height=380, title="PM attendue vs réelle (k€)",
            xaxis_title="PM attendue (k€)", yaxis_title="PM réelle (k€)",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=port_bt["ecart_relatif"], nbinsx=20,
            marker_color="#8b5cf6", opacity=0.7
        ))
        moy = port_bt["ecart_relatif"].mean()
        fig_hist.add_vline(x=moy, line_dash="dash", line_color="red",
                           annotation_text=f"Moy: {moy:.1f}%")
        fig_hist.add_vline(x=0, line_color="black", line_width=1)
        fig_hist.update_layout(
            height=380, title="Distribution des écarts (%)",
            xaxis_title="Écart relatif (%)", yaxis_title="Nb contrats",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("**Analyse par tranche d'âge**")
    df_age = analyse_par_age(port_bt)
    st.dataframe(
        df_age.rename(columns={
            "tranche": "Tranche d'âge", "n_contrats": "N contrats",
            "PM_att": "PM attendue (€)", "PM_reel": "PM réelle (€)",
            "ecart_moy_pct": "Écart moy. (%)", "ecart_total": "Écart total (€)",
            "ecart_pct": "Écart (%)"
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "PM attendue (€)":  st.column_config.NumberColumn(format="%,.0f"),
            "PM réelle (€)":    st.column_config.NumberColumn(format="%,.0f"),
            "Écart total (€)":  st.column_config.NumberColumn(format="%+,.0f"),
            "Écart moy. (%)":   st.column_config.NumberColumn(format="%+.2f"),
            "Écart (%)":        st.column_config.NumberColumn(format="%+.2f"),
        }
    )

with tab3:
    st.markdown("#### Décomposition de l'écart en 3 effets")
    st.markdown("""<div class="formula-box">
    Écart total = Effet taux + Effet mortalité + Effet portefeuille<br>
    Effet taux       : variation de la courbe de taux N → N+1<br>
    Effet mortalité  : décès réels ≠ table (risque longévité)<br>
    Effet portefeuille: rachats réels vs attendus (résiduel)
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    labels  = ["PM\nAttendue", "Effet\nTaux",
               "Effet\nMortalité", "Effet\nPortefeuille", "PM\nRéelle"]
    valeurs = [
        total_att,
        decomp["Effet taux (€)"],
        decomp["Effet mortalité (€)"],
        decomp["Effet portefeuille (€)"],
        total_reel
    ]
    couleurs_bar = [
        "#3b82f6",
        "#ef4444" if decomp["Effet taux (€)"] > 0 else "#10b981",
        "#ef4444" if decomp["Effet mortalité (€)"] > 0 else "#10b981",
        "#f59e0b",
        "#8b5cf6"
    ]

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(
        x=labels,
        y=[v / 1e6 for v in valeurs],
        marker_color=couleurs_bar,
        opacity=0.85,
        hovertemplate="%{x}<br>%{y:.3f} M€<extra></extra>"
    ))
    for i, (lbl, val) in enumerate(zip(labels, valeurs)):
        fig_wf.add_annotation(
            x=lbl, y=val/1e6 + (0.05 if val >= 0 else -0.1),
            text=f"{val/1e6:+.2f}M€",
            showarrow=False, font=dict(size=11)
        )
    fig_wf.update_layout(
        height=420, title="Waterfall — décomposition de l'écart",
        yaxis=dict(title="PM (M€)", gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Effet taux",
                  f"{decomp['Effet taux (€)']/1e6:+.2f} M€",
                  delta=f"{decomp['Effet taux (%)']:+.1f}% de l'écart")
    with col2:
        st.metric("Effet mortalité",
                  f"{decomp['Effet mortalité (€)']/1e6:+.2f} M€",
                  delta=f"{decomp['Effet mortalité (%)']:+.1f}% de l'écart")
    with col3:
        st.metric("Effet portefeuille",
                  f"{decomp['Effet portefeuille (€)']/1e6:+.2f} M€",
                  delta=f"{decomp['Effet portefeuille (%)']:+.1f}% de l'écart")

with tab4:
    st.markdown("#### Rapport de backtesting")
    st.markdown("Ce rapport est produit automatiquement à chaque arrêté. "
                "Il synthétise l'écart, sa décomposition et les recommandations "
                "pour la révision des hypothèses.")
    st.markdown("")

    rapport = rapport_backtesting(
        decomp, n_contrats, int(n_deces), int(n_rachats), int(n_survie),
        taux_N, taux_N1, fact_mort, tx_rachat, tx_rack_att
    )
    st.markdown(f'<div class="rapport-box">{rapport}</div>', unsafe_allow_html=True)

    st.download_button(
        "📥 Télécharger le rapport (.txt)",
        data=rapport,
        file_name="rapport_backtesting_N_N1.txt",
        mime="text/plain",
        use_container_width=True
    )

# ── Formules ────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Rappel des formules — Backtesting N/N+1"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Écart N/N+1**")
        st.latex(r"\text{Écart} = PM^{réelle}_{N+1} - PM^{attendue}_{N+1}")
        st.markdown("**Effet taux**")
        st.latex(r"\text{ET} = PM(taux_{N+1}) - PM(taux_N)")
    with col2:
        st.markdown("**Effet mortalité**")
        st.latex(r"\text{EM} = -(d_{att} - d_{réels}) \times \overline{PM}")
        st.markdown("**Effet portefeuille**")
        st.latex(r"\text{EP} = \text{Écart total} - \text{ET} - \text{EM}")
