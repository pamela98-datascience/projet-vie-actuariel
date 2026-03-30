"""
Page 3 - Provisions Mathématiques (PM)
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)
"""

import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.module1_mortalite import get_table_mortalite, construire_table_complete
from modules.module3_PM import (
    PM_prospective, PM_flux_detail, evolution_PM,
    thiele_verification, valeur_rachat,
    courbe_nelson_siegel, best_estimate, scr_taux
)

st.set_page_config(page_title="Provisions Mathématiques", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .metric-card { background:#f8f9fc; border:1px solid #e2e8f0;
        border-radius:10px; padding:16px 20px; text-align:center; }
    .metric-label { font-size:12px; color:#64748b; font-weight:500;
        text-transform:uppercase; letter-spacing:.05em; }
    .metric-value { font-size:26px; font-weight:700; color:#1e293b; margin:4px 0; }
    .metric-sub   { font-size:12px; color:#94a3b8; }
    .section-tag  { display:inline-block; background:#ede9fe; color:#5b21b6;
        font-size:11px; font-weight:600; padding:2px 10px;
        border-radius:20px; margin-bottom:8px; }
    .formula-box  { background:#f5f3ff; border-left:3px solid #8b5cf6;
        padding:10px 16px; border-radius:0 8px 8px 0;
        font-family:monospace; font-size:13px; color:#4c1d95; }
    .alerte-rouge { background:#fef2f2; border-left:3px solid #ef4444;
        padding:10px 16px; border-radius:0 8px 8px 0;
        font-size:13px; color:#7f1d1d; }
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="section-tag">MODULE 3 · LEÇONS 6–7</span>', unsafe_allow_html=True)
st.title("Provisions Mathématiques (PM)")
st.caption("PM prospective · Thiele · Rachat · Best Estimate S2 · SCR taux")
st.divider()

# ── Chargement ─────────────────────────────────────────────────────────────
df_th = construire_table_complete(get_table_mortalite("TH00-02"))
df_tf = construire_table_complete(get_table_mortalite("TF00-02"))

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres contrat")
    age_0    = st.number_input("Âge à la souscription", 55, 80, 65)
    rente_R  = st.number_input("Rente annuelle (€)", 1000, 50000, 14400, 1000)
    taux_i   = st.slider("Taux technique i (%)", 0.0, 5.0, 2.0, 0.1) / 100
    taux_pen = st.slider("Taux pénalité rachat (%)", 0.0, 10.0, 2.0, 0.5) / 100
    st.divider()
    st.markdown("**S2 — Nelson-Siegel**")
    beta0 = st.slider("β₀ (taux long terme, %)", 0.5, 5.0, 3.0, 0.1) / 100
    beta1 = st.slider("β₁ (pente, %)", -3.0, 0.0, -2.0, 0.1) / 100
    tau   = st.slider("τ (forme)", 1.0, 15.0, 5.0, 0.5)
    st.divider()
    st.info("PM = R × äx+t\nThiele : PM(t+1) = [PM(t)×(1+i)−R]×px")

# ── Calculs de base ────────────────────────────────────────────────────────
PM_0      = PM_prospective(df_th, age_0, rente_R, taux_i)
PM_0_i0   = PM_prospective(df_th, age_0, rente_R, 0.0)
ax        = PM_0 / rente_R
BE_NS     = best_estimate(df_th, age_0, rente_R,
                          lambda t: courbe_nelson_siegel(t, beta0, beta1, tau))
scr       = scr_taux(df_th, age_0, rente_R, taux_i)
VR        = valeur_rachat(PM_0, taux_pen)

# ── KPIs ───────────────────────────────────────────────────────────────────
st.markdown("### Indicateurs clés")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PM à la souscription</div>
        <div class="metric-value">{PM_0:,.0f} €</div>
        <div class="metric-sub">i={taux_i*100:.1f}% | {rente_R:,}€/an</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Économie actualisation</div>
        <div class="metric-value">{PM_0_i0 - PM_0:,.0f} €</div>
        <div class="metric-sub">vs i=0% ({PM_0_i0:,.0f}€)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Best Estimate S2</div>
        <div class="metric-value">{BE_NS:,.0f} €</div>
        <div class="metric-sub">Nelson-Siegel β₀={beta0*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SCR taux (±100bps)</div>
        <div class="metric-value">{scr['SCR taux (€)']:,.0f} €</div>
        <div class="metric-sub">{scr['SCR taux (€)']/PM_0*100:.1f}% de la PM</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Onglets ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 PM prospective", "📈 Évolution & Thiele",
    "💸 Rachat", "🔬 Best Estimate S2"
])

with tab1:
    st.markdown(f"#### PM prospective — Rente {rente_R:,}€/an depuis {age_0} ans")
    st.markdown("""<div class="formula-box">
    PM = R × Σ S(t) × vᵗ<br>
    Chaque flux = rente que l'assureur devra peut-être verser, pondérée par la probabilité de survie et actualisée.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    df_flux = PM_flux_detail(df_th, age_0, rente_R, taux_i)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_flux["Âge"], y=df_flux["Flux R×S(t)×vᵗ (€)"],
        marker_color="#8b5cf6", opacity=0.7, name="Flux R×S(t)×vᵗ",
        hovertemplate="Âge %{x}<br>Flux = %{y:,.0f} €<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_flux["Âge"], y=df_flux["PM cumulée (€)"],
        mode="lines", name="PM cumulée", yaxis="y2",
        line=dict(color="#f59e0b", width=2.5),
        hovertemplate="Âge %{x}<br>PM cumulée = %{y:,.0f} €<extra></extra>"
    ))
    fig.add_hline(y=rente_R, line_dash="dot", line_color="gray",
                  annotation_text=f"Rente brute {rente_R:,}€")
    fig.update_layout(
        height=400,
        yaxis=dict(title="Flux annuel (€)", gridcolor="#f1f5f9"),
        yaxis2=dict(title="PM cumulée (€)", overlaying="y",
                    side="right", showgrid=False),
        xaxis=dict(title="Âge", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    seuil_50 = PM_0 * 0.5
    t_50 = df_flux[df_flux["PM cumulée (€)"] >= seuil_50]["t"].iloc[0]
    st.info(f"50% de la PM est constituée par les flux des **{t_50} premières années** "
            f"(jusqu'à {age_0 + t_50} ans). Essentiel pour la gestion actif-passif.")

    with st.expander("Voir le détail des flux"):
        st.dataframe(df_flux, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("#### Évolution de la PM et vérification de Thiele")
    st.markdown("""<div class="formula-box">
    Récurrence de Thiele : PM(t+1) = [PM(t) × (1+i) − R] × px+t<br>
    La PM diminue chaque année : rentes versées + mortalité + horizon qui se réduit.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    df_evol = evolution_PM(df_th, age_0, rente_R, taux_i)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_evol["Âge"], y=df_evol["PM (€)"],
        mode="lines", fill="tozeroy",
        line=dict(color="#8b5cf6", width=2.5),
        fillcolor="rgba(139,92,246,0.1)", name="PM",
        hovertemplate="Âge %{x}<br>PM = %{y:,.0f} €<extra></extra>"
    ))
    fig2.add_trace(go.Scatter(
        x=df_evol["Âge"], y=df_evol["S(t)"] * PM_0,
        mode="lines", line=dict(color="#10b981", width=2, dash="dot"),
        name="S(t) × PM₀ (référence mortalité)",
        hovertemplate="Âge %{x}<br>S(t)×PM₀ = %{y:,.0f} €<extra></extra>"
    ))
    fig2.add_hline(y=PM_0 * 0.5, line_dash="dot", line_color="orange",
                   annotation_text="50% PM₀")
    fig2.update_layout(
        height=400,
        yaxis=dict(title="PM (€)", gridcolor="#f1f5f9",
                   tickformat=",.0f"),
        xaxis=dict(title="Âge de l'assuré", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Tableau d'évolution**")
    st.dataframe(df_evol, use_container_width=True, hide_index=True,
                 column_config={
                     "PM (€)": st.column_config.NumberColumn(format="%,.0f"),
                     "PM / PM₀ (%)": st.column_config.NumberColumn(format="%.2f"),
                 })

    st.markdown("**Vérification par récurrence de Thiele**")
    df_th_check = thiele_verification(df_th, age_0, rente_R, taux_i)
    st.dataframe(df_th_check, use_container_width=True, hide_index=True)
    st.caption("L'écart résiduel est dû à l'approximation de la récurrence de Thiele — normal en pratique.")

with tab3:
    st.markdown("#### Valeur de rachat")
    st.markdown("""<div class="formula-box">
    Valeur de rachat = PM × (1 − taux pénalité)<br>
    Rachat partiel : seule une fraction de la PM est rachetée, la rente future est réduite.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        t_rachat = st.slider("Année du rachat (t)", 0, 30, 10)
        pct_r    = st.slider("% racheté", 10, 100, 100, 10) / 100

    age_rachat = age_0 + t_rachat
    PM_t = PM_prospective(df_th, min(age_rachat, 105), rente_R, taux_i)
    vr   = valeur_rachat(PM_t, taux_pen, pct_r)

    with col2:
        st.markdown(f"**Âge au rachat : {age_rachat} ans**")
        for k, v in vr.items():
            if "€" in k:
                st.metric(k, f"{v:,.0f} €")
            else:
                st.metric(k, f"{v}")

    rente_apres = rente_R * (1 - pct_r)
    if pct_r < 1.0:
        st.info(f"Après rachat partiel ({pct_r*100:.0f}%) : "
                f"rente résiduelle = **{rente_apres:,.0f} €/an** "
                f"({rente_apres/12:,.0f} €/mois)")

    st.markdown("**Évolution de la valeur de rachat selon l'année**")
    ages_r, vrs = [], []
    for t in range(0, 36):
        a = age_0 + t
        if a > 105: break
        PM_t_ = PM_prospective(df_th, a, rente_R, taux_i)
        vr_   = PM_t_ * (1 - taux_pen)
        ages_r.append(a); vrs.append(vr_)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=ages_r, y=vrs, mode="lines", fill="tozeroy",
        line=dict(color="#10b981", width=2.5),
        fillcolor="rgba(16,185,129,0.1)",
        hovertemplate="Âge %{x}<br>VR = %{y:,.0f} €<extra></extra>"
    ))
    fig3.add_trace(go.Scatter(
        x=ages_r,
        y=[PM_prospective(df_th, a, rente_R, taux_i) for a in ages_r],
        mode="lines", line=dict(color="#8b5cf6", width=2, dash="dot"),
        name="PM",
        hovertemplate="Âge %{x}<br>PM = %{y:,.0f} €<extra></extra>"
    ))
    fig3.add_vline(x=age_rachat, line_dash="dot", line_color="orange",
                   annotation_text=f"t={t_rachat}")
    fig3.update_layout(
        height=360,
        yaxis=dict(title="Valeur (€)", tickformat=",.0f", gridcolor="#f1f5f9"),
        xaxis=dict(title="Âge", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown("#### Best Estimate S2 — courbe des taux")
    st.markdown("""<div class="formula-box">
    BE = Σ R × S(t) × vᵗ(courbe EIOPA)<br>
    Différence vs PM classique : courbe des taux variable vs taux fixe garanti.
    Un taux plus bas → BE plus élevé → plus de provisions.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    scenarios = {
        f"Taux fixe {taux_i*100:.1f}% (PM classique)": lambda t: taux_i,
        "Courbe plate 1%":    lambda t: 0.01,
        "Courbe plate 3%":    lambda t: 0.03,
        f"Nelson-Siegel (β₀={beta0*100:.1f}%, β₁={beta1*100:.1f}%)":
            lambda t: courbe_nelson_siegel(t, beta0, beta1, tau),
    }

    fig4 = go.Figure()
    couleurs = ["#8b5cf6", "#ef4444", "#10b981", "#f59e0b"]

    ages_plot = list(range(age_0, min(age_0 + 41, 106)))
    for (nom, fn), c in zip(scenarios.items(), couleurs):
        vals = [best_estimate(df_th, a, rente_R, fn) for a in ages_plot]
        fig4.add_trace(go.Scatter(
            x=ages_plot, y=vals, mode="lines",
            name=nom, line=dict(color=c, width=2),
            hovertemplate=f"{nom}<br>Âge %{{x}}<br>BE=%{{y:,.0f}}€<extra></extra>"
        ))

    fig4.update_layout(
        height=400,
        yaxis=dict(title="BE / PM (€)", tickformat=",.0f", gridcolor="#f1f5f9"),
        xaxis=dict(title="Âge de l'assuré", gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**SCR taux — choc réglementaire S2 ±100 bps**")
    scr_data = scr_taux(df_th, age_0, rente_R, taux_i)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BE base", f"{scr_data['BE base (€)']:,.0f} €")
    with col2:
        st.metric("BE choc down (−100bps)",
                  f"{scr_data['BE choc down (€)']:,.0f} €",
                  delta=f"+{scr_data['ΔBE down (+100bps)']:,.0f} €",
                  delta_color="inverse")
    with col3:
        st.metric("SCR taux", f"{scr_data['SCR taux (€)']:,.0f} €",
                  help="= max(BE_down − BE_base, BE_base − BE_up)")

    st.markdown("""<div class="alerte-rouge">
    ⚠️ Un choc de −100bps augmente le BE de {:.0f}€ ({:.1f}% de la PM).
    C'est le capital supplémentaire que l'assureur doit détenir sous S2.
    </div>""".format(
        scr_data["ΔBE down (+100bps)"],
        scr_data["ΔBE down (+100bps)"] / PM_0 * 100
    ), unsafe_allow_html=True)

# ── Formules ───────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Rappel des formules — cours ISUP Leçons 6–7"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**PM prospective**")
        st.latex(r"PM_t = R \cdot \sum_{k=0}^{\omega-(x+t)} S_t(k) \cdot v^k")
        st.markdown("**Récurrence de Thiele**")
        st.latex(r"PM_{t+1} = \left[PM_t \cdot (1+i) - R\right] \cdot p_{x+t}")
    with col2:
        st.markdown("**Valeur de rachat**")
        st.latex(r"VR_t = PM_t \times (1 - \tau_{rachat})")
        st.markdown("**SCR taux S2**")
        st.latex(r"SCR_{taux} = \max(BE_{down} - BE_{base},\ BE_{base} - BE_{up})")
