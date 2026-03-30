"""
Page 2 - Tarification
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
    get_table_mortalite, construire_table_complete
)
from modules.module2_tarification import (
    tableau_vt, prime_pure_deces, annuite_actuarielle,
    prime_nette, sensibilite_taux, comparaison_HF
)

st.set_page_config(page_title="Tarification", page_icon="💶", layout="wide")

st.markdown("""
<style>
    .metric-card { background:#f8f9fc; border:1px solid #e2e8f0;
        border-radius:10px; padding:16px 20px; text-align:center; }
    .metric-label { font-size:12px; color:#64748b; font-weight:500;
        text-transform:uppercase; letter-spacing:.05em; }
    .metric-value { font-size:26px; font-weight:700; color:#1e293b; margin:4px 0; }
    .metric-sub   { font-size:12px; color:#94a3b8; }
    .section-tag  { display:inline-block; background:#fef3c7; color:#92400e;
        font-size:11px; font-weight:600; padding:2px 10px;
        border-radius:20px; margin-bottom:8px; }
    .formula-box  { background:#fefce8; border-left:3px solid #eab308;
        padding:10px 16px; border-radius:0 8px 8px 0;
        font-family:monospace; font-size:13px; color:#713f12; }
</style>
""", unsafe_allow_html=True)

st.markdown('<span class="section-tag">MODULE 2 · LEÇONS 4–5</span>', unsafe_allow_html=True)
st.title("Tarification")
st.caption("Prime pure · vᵗ · äx · chargements · sensibilité au taux technique")
st.divider()

# ── Chargement tables ──────────────────────────────────────────────────────
df_th = construire_table_complete(get_table_mortalite("TH00-02"))
df_tf = construire_table_complete(get_table_mortalite("TF00-02"))

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")
    taux_i   = st.slider("Taux technique i (%)", 0.0, 5.0, 2.0, 0.1) / 100
    age_ref  = st.number_input("Âge de l'assuré", 40, 80, 65)
    rente_R  = st.number_input("Rente annuelle (€)", 1000, 50000, 14400, 1000)
    capital  = st.number_input("Capital décès (€)", 10000, 500000, 100000, 10000)
    duree    = st.number_input("Durée garantie décès (ans)", 1, 30, 10)
    alpha    = st.slider("Chargement acquisition α (%)", 0.0, 15.0, 5.0, 0.5) / 100
    st.divider()
    st.markdown("**Rappel cours (leçon 4)**")
    st.info("vᵗ = 1/(1+i)ᵗ\n\nPrime pure rente = R × äx\n\näx = Σ S(t) × vᵗ")

# ── KPIs ───────────────────────────────────────────────────────────────────
pp_deces, _ = prime_pure_deces(df_th, age_ref, int(duree), capital, taux_i)
pp_rente, _ = annuite_actuarielle(df_th, age_ref, taux_i, rente_R)
pn_rente    = prime_nette(pp_rente, alpha)
ax          = pp_rente / rente_R

st.markdown("### Indicateurs clés")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Prime pure décès {int(duree)} ans</div>
        <div class="metric-value">{pp_deces:,.0f} €</div>
        <div class="metric-sub">Capital {capital:,}€ | i={taux_i*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">äx — annuité actuarielle</div>
        <div class="metric-value">{ax:.4f}</div>
        <div class="metric-sub">années actualisées</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Prime pure rente</div>
        <div class="metric-value">{pp_rente:,.0f} €</div>
        <div class="metric-sub">{rente_R:,}€/an | i={taux_i*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Prime nette (α={alpha*100:.0f}%)</div>
        <div class="metric-value">{pn_rente:,.0f} €</div>
        <div class="metric-sub">soit {pn_rente/12:,.0f} €/mois</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Onglets ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Facteur vᵗ", "☠️ Prime pure décès", "🏦 Rente viagère äx", "📊 Sensibilités"
])

with tab1:
    st.markdown("#### Facteur d'actualisation vᵗ = 1/(1+i)ᵗ")
    st.markdown("""<div class="formula-box">
    vᵗ traduit la valeur temps de l'argent : 1€ dans t ans vaut vᵗ € aujourd'hui.
    Plus i est élevé, plus vᵗ est petit → la prime est plus faible.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    taux_dict = {"i=0%": 0.0, "i=1%": 0.01, "i=2%": 0.02, "i=3%": 0.03, "i=5%": 0.05}
    horizons  = list(range(0, 41, 5))
    df_vt     = tableau_vt(taux_dict, horizons)

    fig = go.Figure()
    couleurs = ["#94a3b8", "#10b981", "#3b82f6", "#f59e0b", "#ef4444"]
    for (nom, i), c in zip(taux_dict.items(), couleurs):
        vt_vals = [1/(1+i)**t for t in range(41)]
        lw = 3 if abs(i - taux_i) < 0.005 else 1.5
        fig.add_trace(go.Scatter(
            x=list(range(41)), y=vt_vals, mode="lines",
            name=nom, line=dict(color=c, width=lw),
            hovertemplate=f"{nom}<br>t=%{{x}}<br>vᵗ=%{{y:.4f}}<extra></extra>"
        ))
    fig.add_vline(x=20, line_dash="dot", line_color="gray",
                  annotation_text="20 ans", annotation_position="top")
    fig.update_layout(height=380, xaxis_title="Horizon t (années)",
                      yaxis_title="vᵗ", plot_bgcolor="white",
                      paper_bgcolor="white",
                      yaxis=dict(gridcolor="#f1f5f9"),
                      xaxis=dict(gridcolor="#f1f5f9"),
                      margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_vt, use_container_width=True, hide_index=True)

with tab2:
    st.markdown(f"#### Prime pure — assurance décès temporaire {int(duree)} ans")
    st.markdown("""<div class="formula-box">
    PP = C × Σ(t=0 à n-1) [ tpx × qx+t × v^(t+1) ]<br>
    tpx × qx+t = probabilité de décéder exactement entre t et t+1
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    _, df_flux_dc = prime_pure_deces(df_th, age_ref, int(duree), capital, taux_i)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_flux_dc["Âge"], y=df_flux_dc["Flux (€)"],
        marker_color="#ef4444", opacity=0.7,
        hovertemplate="Âge %{x}<br>Flux = %{y:.2f} €<extra></extra>"
    ))
    fig2.update_layout(height=350, xaxis_title="Âge",
                       yaxis_title="Flux actualisé (€)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(gridcolor="#f1f5f9"),
                       margin=dict(t=20, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**Prime pure unique = {pp_deces:,.2f} €**")
    st.dataframe(df_flux_dc, use_container_width=True, hide_index=True,
                 column_config={
                     "Flux (€)": st.column_config.NumberColumn(format="%.2f"),
                     "tpx": st.column_config.NumberColumn(format="%.6f"),
                 })

with tab3:
    st.markdown(f"#### Rente viagère {rente_R:,} €/an depuis {age_ref} ans")
    st.markdown("""<div class="formula-box">
    äx = Σ(t=0 → ω-x) S(t) × vᵗ<br>
    Prime pure = R × äx = valeur actuelle de toutes les rentes futures
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    _, df_flux_rente = annuite_actuarielle(df_th, age_ref, taux_i, rente_R)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df_flux_rente["Âge"], y=df_flux_rente["Flux (€/an)"],
        marker_color="#8b5cf6", opacity=0.7, name="Flux R×S(t)×vᵗ",
        hovertemplate="Âge %{x}<br>Flux = %{y:,.0f} €<extra></extra>"
    ))
    fig3.add_hline(y=rente_R, line_dash="dot", line_color="gray",
                   annotation_text=f"Rente brute {rente_R:,}€")
    fig3.update_layout(height=360, xaxis_title="Âge",
                       yaxis_title="Flux pondéré (€)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(gridcolor="#f1f5f9"),
                       margin=dict(t=20, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prime pure unique", f"{pp_rente:,.0f} €")
        st.metric("äx (annuité)", f"{ax:.4f} années")
    with col2:
        st.metric(f"Prime nette (α={alpha*100:.0f}%)", f"{pn_rente:,.0f} €")
        st.metric("Mensualité", f"{pn_rente/12:,.0f} €/mois")

    st.markdown("**Comparaison Homme / Femme**")
    df_hf = comparaison_HF(df_th, df_tf, age_ref, rente_R, taux_i)
    st.dataframe(df_hf, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("#### Sensibilité au taux technique")
    st.markdown("""<div class="formula-box">
    Plus i est élevé → vᵗ plus petit → PM plus faible → prime moins chère.<br>
    C'est pourquoi l'ACPR plafonne le taux technique.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    taux_list = [0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    df_sensi  = sensibilite_taux(df_th, age_ref, rente_R, taux_list)
    st.dataframe(df_sensi, use_container_width=True, hide_index=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=[f"{i*100:.1f}%" for i in taux_list],
        y=[sensibilite_taux(df_th, age_ref, rente_R, [i])["Prime pure (€)"].values[0]
           for i in taux_list],
        mode="lines+markers", line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7),
        hovertemplate="i=%{x}<br>PP=%{y:,.0f}€<extra></extra>"
    ))
    fig4.add_vline(x=f"{taux_i*100:.1f}%", line_dash="dot",
                   line_color="orange",
                   annotation_text=f"Taux actuel {taux_i*100:.1f}%")
    fig4.update_layout(height=360, xaxis_title="Taux technique i",
                       yaxis_title="Prime pure (€)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(gridcolor="#f1f5f9"),
                       margin=dict(t=20, b=40))
    st.plotly_chart(fig4, use_container_width=True)

# ── Formules ───────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Rappel des formules — cours ISUP Leçons 4–5"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Actualisation**")
        st.latex(r"v^t = \frac{1}{(1+i)^t}")
        st.markdown("**Prime pure décès temporaire**")
        st.latex(r"PP = C \sum_{t=0}^{n-1} {_tp_x} \cdot q_{x+t} \cdot v^{t+1}")
    with col2:
        st.markdown("**Annuité actuarielle**")
        st.latex(r"\ddot{a}_x = \sum_{t=0}^{\omega-x} S(t) \cdot v^t")
        st.markdown("**Prime nette**")
        st.latex(r"PN = \frac{PP}{1 - \alpha}")
