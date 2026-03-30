"""
Projet Vie Actuariel - Application Streamlit
Pamela Fagla | ISUP M1 Actuariat | 2025-2026

Application de modélisation actuarielle vie :
- Module 1 : Tables de mortalité (TH/TF, lx, qx, ex, S(t))
- Module 2 : Tarification (prime pure, äx, chargements)
- Module 3 : Provisions mathématiques (PM, Thiele, Best Estimate S2)
- Module 4 : Inventaire & résultats (compte technique, PB, ALM)
- Module 5 : Backtesting PM (analyse N/N+1, décomposition écart)
"""

import streamlit as st

st.set_page_config(
    page_title="Projet Vie Actuariel",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .hero { background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
            border-radius: 12px; padding: 32px 40px; color: white; margin-bottom: 24px; }
    .hero h1 { font-size: 28px; font-weight: 700; margin: 0 0 8px 0; }
    .hero p  { font-size: 15px; opacity: 0.85; margin: 0; }
    .hero .badge { display:inline-block; background:rgba(255,255,255,0.2);
                   border-radius:20px; padding:3px 12px; font-size:12px; margin-top:12px; }
    .module-card { border:1px solid #e2e8f0; border-radius:10px; padding:18px 20px;
                   margin-bottom:10px; }
    .module-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

# ── En-tête ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📐 Projet Vie Actuariel</h1>
    <p>Modélisation actuarielle vie complète — Tables · Tarification · PM · Inventaire · Backtesting</p>
    <span class="badge">Python + SAS</span>
    <span class="badge" style="margin-left:8px">ISUP M1 Actuariat 2025–2026</span>
    <span class="badge" style="margin-left:8px">Pamela Fagla</span>
</div>
""", unsafe_allow_html=True)

st.markdown("## Modules du projet")
st.caption("Utilisez la barre latérale pour naviguer entre les modules.")

# ── Modules ──────────────────────────────────────────────────────────────────
modules = [
    (
        "Module 1", "Tables de mortalité",
        "TH/TF 00-02 · qx · lx · dx · ex · loi de survie S(t) · comparaison inter-tables",
        "Leçons 1–3",
        "#dcfce7", "#16a34a"
    ),
    (
        "Module 2", "Tarification",
        "Facteur vᵗ · prime pure décès · annuité äx · chargements α · sensibilité au taux",
        "Leçons 4–5",
        "#dbeafe", "#1d4ed8"
    ),
    (
        "Module 3", "Provisions Mathématiques (PM)",
        "PM prospective · récurrence de Thiele · rachat · Best Estimate S2 · SCR taux ±100bps",
        "Leçons 6–7",
        "#ede9fe", "#6d28d9"
    ),
    (
        "Module 4", "Inventaire & Résultats",
        "Compte technique · résultat financier / mortalité / gestion · PB · duration ALM",
        "Leçons 8–10",
        "#fef9c3", "#92400e"
    ),
    (
        "Module 5", "Backtesting PM",
        "Analyse N/N+1 · écart PM attendue vs réelle · décomposition effet taux / mortalité / portefeuille",
        "Analyse",
        "#fee2e2", "#991b1b"
    ),
]

for num, (label, titre, desc, cours, bg, txt) in enumerate(modules, 1):
    st.markdown(f"""
    <div class="module-card" style="border-left: 4px solid {txt}; background:{bg}10;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:12px; color:#64748b; margin-bottom:3px;">
                    ✅ {label} · {cours}
                </div>
                <div style="font-size:16px; font-weight:600; color:#1e293b;">
                    {titre}
                </div>
                <div style="font-size:13px; color:#475569; margin-top:4px;">
                    {desc}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Métriques ────────────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Modules réalisés", "5 / 5", delta="Complet ✅")
with col_b:
    st.metric("Langages", "Python · SAS")
with col_c:
    st.metric("Tables disponibles", "TH00-02 · TF00-02 · TD88-90")
with col_d:
    st.metric("Contrats simulés", "200 contrats")

st.divider()

# ── Description ──────────────────────────────────────────────────────────────
st.markdown("""
**À propos de ce projet**

Ce projet modélise le cycle complet de l'actuariat retraite collective en Python et SAS :
construction des tables de mortalité TH/TF 00-02, tarification des rentes viagères par la méthode äx,
calcul des provisions mathématiques prospectives avec vérification par la récurrence de Thiele,
inventaire trimestriel avec décomposition du résultat technique, et backtesting N/N+1
avec décomposition de l'écart en effet taux, effet mortalité et effet portefeuille.

Les scripts SAS reproduisent chaque calcul clé dans un environnement professionnel
(DATA step, macros, ODS PDF).

*Pamela Fagla — M1 Actuariat ISUP (Sorbonne Université) — 2025–2026*
""")
