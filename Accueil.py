"""
Projet Vie Actuariel - Application Streamlit
Pamela Fagla | ISUP M1 Actuariat | 2025-2026

Application de modélisation actuarielle vie :
- Module 1 : Tables de mortalité (TH/TF, lx, qx, ex, S(t))
- Module 2 : Tarification (prime pure, chargements)     [à venir]
- Module 3 : Provisions mathématiques (PM, Best Estimate) [à venir]
- Module 4 : Inventaire & résultats (compte technique, PB) [à venir]
- Module 5 : Backtesting PM (analyse N/N+1)              [à venir]
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
    .module-card { border:1px solid #e2e8f0; border-radius:10px; padding:20px;
                   margin-bottom:12px; transition: box-shadow 0.2s; }
    .module-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .status-done { color:#16a34a; font-weight:600; }
    .status-todo { color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📐 Projet Vie Actuariel</h1>
    <p>Modélisation actuarielle vie — Tables de mortalité · Tarification · PM · Backtesting</p>
    <span class="badge">Python + SAS</span>
    <span class="badge" style="margin-left:8px">ISUP M1 Actuariat 2025–2026</span>
    <span class="badge" style="margin-left:8px">Pamela Fagla</span>
</div>
""", unsafe_allow_html=True)

st.markdown("## Modules du projet")
st.caption("Utilisez la barre latérale pour naviguer entre les modules.")

modules = [
    ("✅", "Module 1", "Tables de mortalité",
     "TH/TF · qx · lx · dx · ex · loi de survie S(t) · comparaison inter-tables",
     "Leçons 1–3", "done"),
    ("🔜", "Module 2", "Tarification",
     "Prime pure · prime nette · chargements · taux garanti réglementaire · sensibilités",
     "Leçons 4–5", "todo"),
    ("🔜", "Module 3", "Provisions mathématiques (PM)",
     "Best Estimate · PM prospective · rachat · réduction · courbe des taux Nelson-Siegel · SCR vie",
     "Leçons 6–7", "todo"),
    ("🔜", "Module 4", "Inventaire & résultats",
     "Compte technique · résultat mortalité/financier/gestion · participation aux bénéfices",
     "Leçons 8–10", "todo"),
    ("🔜", "Module 5", "Backtesting PM",
     "Comparaison N/N+1 · analyse écarts · revue hypothèses mortalité et taux",
     "Analyse", "todo"),
]

for icon, num, titre, desc, cours, status in modules:
    col1, col2 = st.columns([5, 1])
    with col1:
        classe = "status-done" if status == "done" else "status-todo"
        st.markdown(f"""
        <div class="module-card">
            <div style="font-size:13px;color:#64748b;margin-bottom:4px">{cours}</div>
            <div style="font-size:17px;font-weight:600;color:#1e293b">{icon} {num} — {titre}</div>
            <div style="font-size:13px;color:#475569;margin-top:4px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Modules réalisés", "1 / 5")
with col_b:
    st.metric("Langages", "Python · SAS")
with col_c:
    st.metric("Tables disponibles", "TH00-02 · TF00-02 · TD88-90")

st.divider()
st.markdown("""
**Utilisation :** Ce projet a été construit pour approfondir les notions du cours d'assurance vie (ISUP M1 Actuariat) 
et démontrer une maîtrise opérationnelle des calculs actuariels vie en Python et SAS.  
Les scripts SAS (`/sas/`) permettent de reproduire chaque calcul dans un environnement professionnel.
""")
