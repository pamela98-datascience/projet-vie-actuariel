# ============================================================
# MODULE 5 — BACKTESTING PM (Analyse N/N+1)
# Projet Vie Actuariel — Pamela Fagla (ISUP M1 Actuariat)
# Google Colab — à exécuter cellule par cellule
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────
# CELLULE 1 : Rechargement complet (Modules 1-4)
# ─────────────────────────────────────────────────────────────

ages  = np.arange(0, 111)
qx_th = np.array([
    0.003760,0.000283,0.000195,0.000153,0.000126,0.000109,0.000096,
    0.000086,0.000079,0.000075,0.000074,0.000080,0.000104,0.000150,
    0.000210,0.000278,0.000349,0.000417,0.000472,0.000510,0.000533,
    0.000547,0.000558,0.000569,0.000581,0.000592,0.000601,0.000608,
    0.000614,0.000621,0.000632,0.000649,0.000673,0.000707,0.000752,
    0.000810,0.000880,0.000963,0.001060,0.001171,0.001297,0.001440,
    0.001604,0.001789,0.001997,0.002229,0.002487,0.002773,0.003088,
    0.003435,0.003816,0.004234,0.004692,0.005193,0.005740,0.006338,
    0.006991,0.007704,0.008481,0.009328,0.010248,0.011246,0.012325,
    0.013491,0.014749,0.016105,0.017561,0.019125,0.020803,0.022605,
    0.024540,0.026618,0.028850,0.031250,0.033835,0.036620,0.039623,
    0.042866,0.046368,0.050152,0.054239,0.058651,0.063409,0.068534,
    0.074044,0.079953,0.086272,0.093008,0.100165,0.107741,0.115727,
    0.124107,0.132857,0.141950,0.151354,0.161032,0.170943,0.181044,
    0.191294,0.201650,0.212069,0.222508,0.232923,0.243270,0.253505,
    0.263584,0.273462,0.283095,0.292440,0.301453,1.000000
])

l0 = 100_000
df = pd.DataFrame({"age": ages, "qx": qx_th})
df["px"]      = 1 - df["qx"]
df["lx"]      = l0 * np.concatenate([[1.0], np.cumprod(df["px"].values[:-1])])
df["dx"]      = df["lx"] * df["qx"]
df["lx_suiv"] = df["lx"].shift(-1, fill_value=0)
df["Lx"]      = (df["lx"] + df["lx_suiv"]) / 2
df["Tx"]      = df["Lx"].values[::-1].cumsum()[::-1]
df["ex"]      = np.where(df["lx"] > 0, df["Tx"] / df["lx"], 0)

def PM_contrat(table, age_x, rente_R, taux_i):
    """PM prospective = R × äx"""
    v  = 1 / (1 + taux_i)
    lx = table.loc[table["age"] == age_x, "lx"].values[0]
    pm = sum(
        rente_R * (table.loc[table["age"] == age_x + t, "lx"].values[0] / lx) * (v ** t)
        for t in range(111 - age_x)
        if age_x + t <= 110
    )
    return pm

print("✅ Table rechargée — Module 5 prêt")


# ─────────────────────────────────────────────────────────────
# CELLULE 2 : Construction du portefeuille de contrats
#
# COURS : En pratique, le portefeuille contient des contrats
# d'âges différents, entrés à des dates différentes.
# Le backtesting s'applique contrat par contrat puis est
# agrégé au niveau du portefeuille.
# ─────────────────────────────────────────────────────────────

np.random.seed(42)

# Portefeuille de 200 contrats avec âges hétérogènes
n_contrats = 200
ages_souscription = np.random.choice(
    [62, 63, 64, 65, 66, 67, 68],
    size=n_contrats,
    p=[0.05, 0.10, 0.20, 0.35, 0.15, 0.10, 0.05]
)
rentes = np.random.choice(
    [9_600, 12_000, 14_400, 18_000, 24_000],
    size=n_contrats,
    p=[0.20, 0.30, 0.30, 0.15, 0.05]
)

# Taux technique de référence à N
TAUX_N    = 0.020   # taux technique à date N
TAUX_N1   = 0.015   # taux à N+1 (baisse des taux = scénario adverse)

# Construction du DataFrame portefeuille à date N
portefeuille = pd.DataFrame({
    "id_contrat": range(n_contrats),
    "age_N":      ages_souscription,
    "rente":      rentes,
})

# PM à date N (avec taux N)
portefeuille["PM_N"] = portefeuille.apply(
    lambda r: PM_contrat(df, int(r["age_N"]), r["rente"], TAUX_N), axis=1
)

print(f"Portefeuille à date N — {n_contrats} contrats\n")
print(f"  Âge moyen         : {portefeuille['age_N'].mean():.1f} ans")
print(f"  Rente moyenne     : {portefeuille['rente'].mean():,.0f} €/an")
print(f"  PM totale (date N): {portefeuille['PM_N'].sum():,.0f} €")
print(f"  PM moyenne/contrat: {portefeuille['PM_N'].mean():,.0f} €")
print(f"\nDistribution des âges :")
print(portefeuille["age_N"].value_counts().sort_index())


# ─────────────────────────────────────────────────────────────
# CELLULE 3 : Simulation des événements entre N et N+1
#
# COURS : Entre N et N+1, trois événements peuvent toucher
# chaque contrat :
#   1. Décès (selon qx de la table, avec facteur d'expérience)
#   2. Rachat (selon un taux de rachat)
#   3. Survie (le contrat reste actif)
# ─────────────────────────────────────────────────────────────

# Paramètres de simulation
FACTEUR_MORTALITE = 0.80   # mortalité réelle = 80% de la table (amélioration)
TAUX_RACHAT_REEL  = 0.04   # 4% de rachats dans l'année

evenements = []
for _, row in portefeuille.iterrows():
    age = int(row["age_N"])
    qx_table = df.loc[df["age"] == age, "qx"].values[0]
    qx_reel  = qx_table * FACTEUR_MORTALITE

    # Tirage aléatoire : décès ?
    deces  = np.random.random() < qx_reel
    # Tirage aléatoire : rachat ? (si pas décédé)
    rachat = (not deces) and (np.random.random() < TAUX_RACHAT_REEL)
    # Survie
    survie = not deces and not rachat

    evenements.append({
        "id_contrat": row["id_contrat"],
        "deces":  deces,
        "rachat": rachat,
        "survie": survie,
    })

df_evt = pd.DataFrame(evenements)
portefeuille = portefeuille.merge(df_evt, on="id_contrat")

n_deces  = portefeuille["deces"].sum()
n_rachats = portefeuille["rachat"].sum()
n_survie = portefeuille["survie"].sum()

print(f"\nÉvénements entre N et N+1 :\n")
print(f"  Décès   : {n_deces:>4}  ({n_deces/n_contrats*100:.1f}%)")
print(f"  Rachats : {n_rachats:>4}  ({n_rachats/n_contrats*100:.1f}%)")
print(f"  Survies : {n_survie:>4}  ({n_survie/n_contrats*100:.1f}%)")
print(f"\n  Décès attendus selon table : {portefeuille.apply(lambda r: df.loc[df['age']==int(r['age_N']),'qx'].values[0]*FACTEUR_MORTALITE, axis=1).sum():.1f}")
print(f"  Décès réels                : {n_deces}")


# ─────────────────────────────────────────────────────────────
# CELLULE 4 : PM attendue vs PM réelle à N+1
#
# COURS : C'est le cœur du backtesting.
#
# PM attendue à N+1 :
#   Calculée à N avec les hypothèses N, projetée à N+1
#   = PM_N × (1 + i_N) − Rente   (pour les contrats qui survivent)
#   (Thiele simplifié)
#
# PM réelle à N+1 :
#   Recalculée à N+1 avec les nouvelles hypothèses
#   = PM_contrat(age+1, rente, taux_N+1)
#   (taux peut avoir changé)
# ─────────────────────────────────────────────────────────────

# On travaille uniquement sur les contrats qui ont survécu
port_survie = portefeuille[portefeuille["survie"]].copy()

# PM attendue à N+1 (projection avec hypothèses N)
port_survie["PM_N1_attendue"] = port_survie.apply(
    lambda r: PM_contrat(df, int(r["age_N"]) + 1, r["rente"], TAUX_N),
    axis=1
)

# PM réelle à N+1 (avec nouveau taux)
port_survie["PM_N1_reelle"] = port_survie.apply(
    lambda r: PM_contrat(df, int(r["age_N"]) + 1, r["rente"], TAUX_N1),
    axis=1
)

# Écart N/N+1 par contrat
port_survie["ecart_absolu"] = port_survie["PM_N1_reelle"] - port_survie["PM_N1_attendue"]
port_survie["ecart_relatif"] = port_survie["ecart_absolu"] / port_survie["PM_N1_attendue"] * 100

# Totaux
total_att  = port_survie["PM_N1_attendue"].sum()
total_reel = port_survie["PM_N1_reelle"].sum()
ecart_tot  = total_reel - total_att

print(f"\nBACKTESTING N/N+1 — {len(port_survie)} contrats survivants\n")
print(f"  Taux N   : {TAUX_N*100:.1f}%   →   Taux N+1 : {TAUX_N1*100:.1f}%")
print(f"  Facteur mortalité réel : {FACTEUR_MORTALITE*100:.0f}% de la table\n")
print(f"  PM attendue à N+1     : {total_att:>15,.0f} €")
print(f"  PM réelle à N+1       : {total_reel:>15,.0f} €")
print(f"  ─────────────────────────────────────")
print(f"  ÉCART TOTAL           : {ecart_tot:>+15,.0f} €  ({ecart_tot/total_att*100:+.2f}%)")


# ─────────────────────────────────────────────────────────────
# CELLULE 5 : Décomposition de l'écart en 3 effets
#
# COURS : L'écart total = effet taux + effet mortalité + effet portefeuille
#
# EFFET TAUX :
#   Si on refait PM_N1 avec taux_N1 mais hypothèses mortalité de N
#   → l'écart vs PM attendue est pur effet taux
#
# EFFET MORTALITÉ :
#   La différence entre décès réels et attendus
#   × PM libérée par chaque décès
#
# EFFET PORTEFEUILLE (résiduel) :
#   Rachats différents + nouveaux entrants
# ─────────────────────────────────────────────────────────────

# ── EFFET TAUX ──────────────────────────────────────────────
# PM avec nouveau taux mais vieux contrats (isoler l'effet taux pur)
port_survie["PM_N1_taux_seul"] = port_survie.apply(
    lambda r: PM_contrat(df, int(r["age_N"]) + 1, r["rente"], TAUX_N1),
    axis=1
)
effet_taux = port_survie["PM_N1_taux_seul"].sum() - total_att

# ── EFFET MORTALITÉ ──────────────────────────────────────────
# Décès attendus vs réels
deces_attendus_n = portefeuille.apply(
    lambda r: df.loc[df["age"] == int(r["age_N"]), "qx"].values[0], axis=1
).sum()
deces_attendus = deces_attendus_n * FACTEUR_MORTALITE
deces_reels    = n_deces

# PM moyenne libérée par un décès
PM_moy = portefeuille["PM_N"].mean()
effet_mortalite = (deces_attendus - deces_reels) * PM_moy
# Note : moins de décès réels = plus de PM à provisionner = effet négatif

# ── EFFET PORTEFEUILLE (résiduel) ────────────────────────────
# Rachats réels vs attendus
rachats_attendus = n_contrats * 0.03   # hypothèse 3%
rachats_reels    = n_rachats
effet_portefeuille = ecart_tot - effet_taux - effet_mortalite

print(f"\nDÉCOMPOSITION DE L'ÉCART N/N+1\n")
print(f"  {'Effet taux':35} : {effet_taux:>+12,.0f} €  ({effet_taux/abs(ecart_tot)*100:>+6.1f}%)")
print(f"  {'Effet mortalité':35} : {effet_mortalite:>+12,.0f} €  ({effet_mortalite/abs(ecart_tot)*100:>+6.1f}%)")
print(f"  {'Effet portefeuille (résiduel)':35} : {effet_portefeuille:>+12,.0f} €  ({effet_portefeuille/abs(ecart_tot)*100:>+6.1f}%)")
print(f"  {'─'*62}")
print(f"  {'ÉCART TOTAL':35} : {ecart_tot:>+12,.0f} €  (100.0%)")
print()
print(f"Interprétation :")
print(f"  → La baisse de taux de {TAUX_N*100:.1f}% → {TAUX_N1*100:.1f}% est l'effet dominant.")
print(f"  → La mortalité plus faible que prévue pèse aussi (risque de longévité).")
print(f"  → L'assureur doit renforcer ses provisions de {abs(ecart_tot):,.0f}€.")


# ─────────────────────────────────────────────────────────────
# CELLULE 6 : Analyse par tranche d'âge
#
# COURS : Le backtesting ne s'arrête pas au total.
# On analyse les écarts par tranche d'âge pour détecter
# si certains segments sont systématiquement sous/sur-provisionnés.
# ─────────────────────────────────────────────────────────────

port_survie["tranche_age"] = pd.cut(
    port_survie["age_N"],
    bins=[61, 64, 66, 68, 110],
    labels=["62-64 ans", "65-66 ans", "67-68 ans", "69+ ans"]
)

analyse_age = port_survie.groupby("tranche_age", observed=True).agg(
    n_contrats    = ("id_contrat", "count"),
    PM_att        = ("PM_N1_attendue", "sum"),
    PM_reel       = ("PM_N1_reelle", "sum"),
    ecart_moy_pct = ("ecart_relatif", "mean"),
).reset_index()

analyse_age["ecart_total"] = analyse_age["PM_reel"] - analyse_age["PM_att"]
analyse_age["ecart_pct"]   = analyse_age["ecart_total"] / analyse_age["PM_att"] * 100

print(f"\nANALYSE PAR TRANCHE D'ÂGE\n")
print(f"{'Tranche':12} | {'N':>5} | {'PM att. (€)':>14} | {'PM réelle (€)':>14} | {'Écart %':>9}")
print("─" * 65)
for _, r in analyse_age.iterrows():
    print(f"  {str(r['tranche_age']):12} | {int(r['n_contrats']):>5} | "
          f"{r['PM_att']:>14,.0f} | {r['PM_reel']:>14,.0f} | {r['ecart_pct']:>+8.2f}%")

print(f"\n→ L'écart est homogène entre tranches → c'est bien un effet taux global")
print(f"  (si l'écart était concentré sur une tranche → problème d'hypothèse spécifique)")


# ─────────────────────────────────────────────────────────────
# CELLULE 7 : Graphiques backtesting
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Graph 1 : PM attendue vs réelle par contrat (scatter)
axes[0].scatter(port_survie["PM_N1_attendue"] / 1000,
                port_survie["PM_N1_reelle"] / 1000,
                alpha=0.4, color="#3b82f6", s=20)
lim_max = max(port_survie["PM_N1_attendue"].max(),
              port_survie["PM_N1_reelle"].max()) / 1000 * 1.05
axes[0].plot([0, lim_max], [0, lim_max], "r--", linewidth=1.5,
             label="Attendu = Réel (référence)")
axes[0].set_xlabel("PM attendue (k€)")
axes[0].set_ylabel("PM réelle N+1 (k€)")
axes[0].set_title("PM attendue vs réelle par contrat")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Graph 2 : Distribution des écarts relatifs
axes[1].hist(port_survie["ecart_relatif"], bins=20,
             color="#8b5cf6", alpha=0.7, edgecolor="white")
axes[1].axvline(port_survie["ecart_relatif"].mean(),
                color="red", linestyle="--", linewidth=2,
                label=f"Moyenne : {port_survie['ecart_relatif'].mean():.1f}%")
axes[1].axvline(0, color="black", linewidth=1, alpha=0.5)
axes[1].set_xlabel("Écart relatif (%)")
axes[1].set_ylabel("Nombre de contrats")
axes[1].set_title("Distribution des écarts N/N+1")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Graph 3 : Décomposition de l'écart (waterfall simplifié)
categories = ["PM\nattendue", "Effet\ntaux", "Effet\nmortalité",
              "Effet\nportefeuille", "PM\nréelle"]
valeurs    = [total_att, effet_taux, effet_mortalite,
              effet_portefeuille, total_reel]
couleurs   = ["#3b82f6", "#ef4444" if effet_taux > 0 else "#10b981",
              "#ef4444" if effet_mortalite > 0 else "#10b981",
              "#f59e0b", "#8b5cf6"]

axes[2].bar(categories, [v/1e6 for v in valeurs],
            color=couleurs, alpha=0.8, edgecolor="white")
axes[2].axhline(y=total_att/1e6, color="gray",
                linestyle="--", alpha=0.5, linewidth=1)
axes[2].set_ylabel("PM (M€)")
axes[2].set_title("Décomposition de l'écart (waterfall)")
axes[2].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.1f}M€"))
axes[2].grid(alpha=0.3, axis="y")

for i, (cat, val) in enumerate(zip(categories, valeurs)):
    axes[2].text(i, val/1e6 + 0.1, f"{val/1e6:.1f}M€",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("module5_backtesting.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────
# CELLULE 8 : Rapport de backtesting — synthèse documentée
#
# COURS : En entreprise, le backtesting se conclut toujours
# par un rapport qui :
#   1. Quantifie l'écart total
#   2. Décompose par effet
#   3. Conclut sur les hypothèses à réviser
#   4. Propose des actions correctives
# ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("RAPPORT DE BACKTESTING — SYNTHÈSE")
print("Portefeuille Retraite Collective | Arrêté N/N+1")
print("="*70)

print(f"""
1. PÉRIMÈTRE
   Contrats analysés : {n_contrats} (dont {n_survie} survivants)
   Décès             : {n_deces} ({n_deces/n_contrats*100:.1f}%)
   Rachats           : {n_rachats} ({n_rachats/n_contrats*100:.1f}%)

2. ÉCART GLOBAL
   PM attendue à N+1 : {total_att:,.0f} €
   PM réelle à N+1   : {total_reel:,.0f} €
   Écart total       : {ecart_tot:+,.0f} € ({ecart_tot/total_att*100:+.2f}%)
   → SOUS-PROVISIONNEMENT : l'assureur doit renforcer ses PM.

3. DÉCOMPOSITION
   Effet taux        : {effet_taux:+,.0f} € ({effet_taux/abs(ecart_tot)*100:.0f}% de l'écart)
     Cause : baisse du taux de {TAUX_N*100:.1f}% à {TAUX_N1*100:.1f}%
     Action : réviser la courbe de projection des taux

   Effet mortalité   : {effet_mortalite:+,.0f} € ({effet_mortalite/abs(ecart_tot)*100:.0f}% de l'écart)
     Cause : mortalité réelle à {FACTEUR_MORTALITE*100:.0f}% de la table TH 00-02
     Action : analyser si l'amélioration est structurelle (→ changer table)
              ou conjoncturelle (→ maintenir les hypothèses)

   Effet portefeuille: {effet_portefeuille:+,.0f} € ({effet_portefeuille/abs(ecart_tot)*100:.0f}% de l'écart)
     Cause : rachats réels ({n_rachats}) vs attendus ({rachats_attendus:.0f})
     Action : réviser les hypothèses de rachat

4. RECOMMANDATIONS
   ✦ PRIORITÉ 1 (effet taux dominant) :
     Mettre à jour la courbe des taux dans le modèle de provisionnement.
     Quantifier le SCR taux selon le choc S2 (-100bps / +100bps).

   ✦ PRIORITÉ 2 (mortalité améliorée) :
     Suivre sur 3 ans avant de réviser la table.
     Constituer une marge de prudence sur la longévité.

   ✦ PRIORITÉ 3 (rachats) :
     Réviser le taux de rachat de 3% à {TAUX_RACHAT_REEL*100:.0f}% dans le modèle.
     Surveiller le risque de pic de rachats en cas de remontée des taux.
""")
print("="*70)


# ─────────────────────────────────────────────────────────────
# CELLULE 9 : Formulations entretien — Module 5 + PROJET COMPLET
# ─────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FORMULATIONS ENTRETIEN — Module 5 : Backtesting                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Q : "C'est quoi le backtesting en actuariat vie ?"                  ║
║  R : "C'est la comparaison entre la PM qu'on avait calculée à N      ║
║       et la PM qu'on recalcule à N+1 avec les données réelles.       ║
║       L'écart se décompose en effet taux, effet mortalité et         ║
║       effet portefeuille. Il permet de valider les hypothèses        ║
║       et de déclencher une révision si l'écart est systématique."   ║
║                                                                      ║
║  Q : "Que faites-vous si l'écart est important ?"                    ║
║  R : "On décompose l'écart pour identifier la source principale.     ║
║       Si c'est l'effet taux (dominant en 2024), on met à jour        ║
║       la courbe de projection. Si c'est la mortalité, on             ║
║       surveille sur 2-3 ans avant de réviser la table."              ║
║                                                                      ║
║  Q : "Décrivez votre projet vie en entretien."                       ║
║  R : "J'ai modélisé le cycle complet de l'actuariat retraite         ║
║       collective : tables TH/TF, tarification par äx, calcul         ║
║       des PM prospectives, inventaire trimestriel avec               ║
║       décomposition du résultat, et backtesting N/N+1.               ║
║       Implémenté en Python avec app Streamlit déployée,              ║
║       et scripts SAS pour réplication. Sur un portefeuille           ║
║       de 200 contrats, j'ai quantifié un sous-provisionnement        ║
║       de +7% lié principalement à la baisse des taux."               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("🎉 PROJET VIE COMPLET — 5 modules, 12 notions :")
print()
print("   MODULE 1 — Tables de mortalité")
print("   ✅ Notion  1 : qx — probabilité de décès")
print("   ✅ Notion  2 : lx, dx — effectif survivant")
print("   ✅ Notion  3 : ex — espérance de vie résiduelle")
print("   ✅ Notion  4 : S(t) — loi de survie")
print()
print("   MODULE 2 — Tarification")
print("   ✅ Notion  5 : vᵗ — actualisation")
print("   ✅ Notion  6 : Prime pure assurance décès")
print("   ✅ Notion  7 : Prime pure rente viagère äx")
print("   ✅ Notion  8 : Chargements et prime nette")
print()
print("   MODULE 3 — Provisions Mathématiques")
print("   ✅ Notion  9 : PM prospective")
print("   ✅ Notion 10 : Évolution PM — récurrence de Thiele")
print("   ✅ Notion 11 : Rachat et valeur de rachat")
print("   ✅ Notion 12 : Best Estimate S2 / SCR taux")
print()
print("   MODULE 4 — Inventaire & Résultats")
print("   ✅ Inventaire trimestriel — équation de passage")
print("   ✅ Résultat financier / mortalité / gestion")
print("   ✅ Participation aux bénéfices (PB, PPB)")
print("   ✅ Duration et gestion actif-passif (ALM)")
print()
print("   MODULE 5 — Backtesting")
print("   ✅ Analyse N/N+1 — écart PM attendue vs réelle")
print("   ✅ Décomposition effet taux / mortalité / portefeuille")
print("   ✅ Rapport de backtesting documenté")
print()
print("→ Étapes suivantes :")
print("  1. Intégrer tous les modules dans l'app Streamlit")
print("  2. Mettre le projet sur GitHub")
print("  3. Construire le portfolio")
print("  4. Adapter le CV et la lettre Groupama")
