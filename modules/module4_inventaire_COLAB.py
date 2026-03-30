# ============================================================
# MODULE 4 — INVENTAIRE & RÉSULTATS (Leçons 8-10)
# Projet Vie Actuariel — Pamela Fagla (ISUP M1 Actuariat)
# Google Colab — à exécuter cellule par cellule
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────
# CELLULE 1 : Rechargement table + fonctions Modules 1-3
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

def PM_prospective(table, age_x, rente_R, taux_i):
    """PM = R × äx = Σ S(t) × vᵗ × R"""
    v   = 1 / (1 + taux_i)
    lx  = table.loc[table["age"] == age_x, "lx"].values[0]
    pm  = 0.0
    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110: break
        St = table.loc[table["age"] == age_t, "lx"].values[0] / lx
        pm += rente_R * St * (v ** t)
    return pm

# Paramètres du contrat de référence
AGE_0       = 65       # âge à la souscription
RENTE_R     = 14_400   # 1 200 €/mois
TAUX_I      = 0.02     # taux technique garanti
N_ASSURES   = 1_000    # portefeuille de 1 000 assurés

PM_0 = PM_prospective(df, AGE_0, RENTE_R, TAUX_I)
print(f"✅ Modules 1-3 rechargés")
print(f"PM initiale par assuré : {PM_0:,.2f} €")
print(f"PM portefeuille ({N_ASSURES} assurés) : {PM_0 * N_ASSURES:,.2f} €")


# ═════════════════════════════════════════════════════════════
# INVENTAIRE TRIMESTRIEL — SIMULATION SUR 10 ANS
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 2 : Paramètres du portefeuille et hypothèses
#
# COURS : L'inventaire compare les réalisations (ce qui s'est
# passé) aux hypothèses (ce qu'on avait prévu).
# L'écart = le résultat technique.
# ─────────────────────────────────────────────────────────────

# Hypothèses de tarification (base du calcul des PM)
hyp = {
    "taux_i":          0.02,    # taux technique garanti
    "taux_gestion":    0.008,   # chargements de gestion (β = 0.8%/an)
    "taux_rachat":     0.03,    # taux de rachat annuel (3% sortent)
}

# Scénarios réels (ce qui se passe vraiment)
# On simule 3 années avec des réalisations différentes
scenarios_annuels = [
    # Année 1 : bonne année (mortalité faible, rendement élevé)
    {"annee": 1, "rendement_reel": 0.035, "tx_mortalite_reel": 0.85,
     "frais_reels_pct": 0.006, "tx_rachat_reel": 0.02},
    # Année 2 : année normale
    {"annee": 2, "rendement_reel": 0.022, "tx_mortalite_reel": 1.00,
     "frais_reels_pct": 0.008, "tx_rachat_reel": 0.03},
    # Année 3 : mauvaise année (longévité + taux bas)
    {"annee": 3, "rendement_reel": 0.012, "tx_mortalite_reel": 0.75,
     "frais_reels_pct": 0.009, "tx_rachat_reel": 0.04},
]

print("\nParamètres du portefeuille :")
print(f"  Nombre d'assurés initial : {N_ASSURES:,}")
print(f"  Âge moyen au départ      : {AGE_0} ans")
print(f"  Rente annuelle           : {RENTE_R:,} €/an")
print(f"  Taux technique garanti   : {TAUX_I*100}%")
print(f"  PM portefeuille initiale : {PM_0*N_ASSURES:,.0f} €")


# ─────────────────────────────────────────────────────────────
# CELLULE 3 : Simulation de l'inventaire année par année
#
# COURS : L'équation de l'inventaire :
#   PM_fin = PM_deb + Intérêts − Rentes − Rachats ± Δ mortalité + PB
#
# Le résultat technique se décompose en :
#   Rés. financier  = (rendement réel − taux garanti) × PM_deb
#   Rés. mortalité  = (décès attendus − décès réels) × PM_par_tête
#   Rés. gestion    = chargements perçus − frais réels
# ─────────────────────────────────────────────────────────────

def inventaire_annuel(PM_deb, n_assures, age_moy, annee_data, hyp, df_table, rente_R, taux_i):
    """
    Calcule l'inventaire d'une année et décompose le résultat.

    Retourne un dict avec tous les postes du compte de résultat.
    """
    r_reel  = annee_data["rendement_reel"]
    tx_mort = annee_data["tx_mortalite_reel"]   # ratio réel/attendu
    frais_r = annee_data["frais_reels_pct"]
    tx_rach = annee_data["tx_rachat_reel"]

    # ── 1. PM par tête et PM portefeuille ──────────────────
    PM_par_tete = PM_deb / n_assures

    # ── 2. Décès ──────────────────────────────────────────
    qx_attendu  = df_table.loc[df_table["age"] == age_moy, "qx"].values[0]
    deces_att   = n_assures * qx_attendu            # attendus (table)
    deces_reels = deces_att * tx_mort               # réels (scénario)

    # ── 3. Rachats ─────────────────────────────────────────
    n_rachats   = n_assures * tx_rach
    taux_penali = hyp["taux_rachat"]
    VR_unitaire = PM_par_tete * (1 - taux_penali)
    rachats_tot = n_rachats * VR_unitaire
    penalites   = n_rachats * PM_par_tete * taux_penali

    # ── 4. Assurés restants en fin d'année ─────────────────
    n_fin       = n_assures - deces_reels - n_rachats
    n_fin       = max(n_fin, 0)

    # ── 5. Rentes versées ──────────────────────────────────
    # On verse à ceux qui sont en vie au début (approx milieu d'année)
    n_moyen    = (n_assures + n_fin) / 2
    rentes_tot = n_moyen * rente_R

    # ── 6. Produits financiers ─────────────────────────────
    prod_financiers_reels = PM_deb * r_reel
    prod_financiers_att   = PM_deb * taux_i
    res_financier         = prod_financiers_reels - prod_financiers_att

    # ── 7. Résultat mortalité ──────────────────────────────
    # Si moins de décès réels → assureur doit provisionner plus → perte
    # (risque de longévité pour les rentes)
    surplus_deces  = deces_att - deces_reels   # > 0 si moins de décès que prévu
    res_mortalite  = -surplus_deces * PM_par_tete
    #                 ↑ signe négatif : moins de décès = perte pour rentes

    # ── 8. Résultat gestion ────────────────────────────────
    chargements_percus = PM_deb * hyp["taux_gestion"]
    frais_reels_eur    = PM_deb * frais_r
    res_gestion        = chargements_percus - frais_reels_eur

    # ── 9. Résultat technique total ────────────────────────
    res_total = res_financier + res_mortalite + res_gestion

    # ── 10. Participation aux bénéfices (PB) ───────────────
    # Minimum légal : 85% rés. financier + 90% rés. technique
    PB_min = max(0, 0.85 * res_financier) + max(0, 0.90 * (res_mortalite + res_gestion))
    PB_tot = PB_min  # on prend le minimum légal ici

    # ── 11. PM de fin d'année ──────────────────────────────
    age_fin    = age_moy + 1
    PM_par_tete_fin = PM_prospective(df_table, age_fin, rente_R, taux_i)
    PM_fin_att = PM_par_tete_fin * n_fin

    return {
        "n_debut": n_assures, "n_fin": round(n_fin),
        "deces_att": round(deces_att, 1), "deces_reels": round(deces_reels, 1),
        "n_rachats": round(n_rachats, 1),
        "PM_deb": PM_deb, "PM_fin": PM_fin_att,
        "rentes_versees": round(rentes_tot, 0),
        "prod_fin_reels": round(prod_financiers_reels, 0),
        "prod_fin_att": round(prod_financiers_att, 0),
        "res_financier": round(res_financier, 0),
        "res_mortalite": round(res_mortalite, 0),
        "res_gestion": round(res_gestion, 0),
        "res_total": round(res_total, 0),
        "PB": round(PB_tot, 0),
        "penalites": round(penalites, 0),
    }


# ─────────────────────────────────────────────────────────────
# CELLULE 4 : Exécution de l'inventaire sur 3 ans
# ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("INVENTAIRE TECHNIQUE — PORTEFEUILLE RETRAITE COLLECTIVE")
print(f"Rente {RENTE_R:,}€/an | {N_ASSURES:,} assurés | Âge moyen départ {AGE_0} ans")
print("="*70)

PM_courant  = PM_0 * N_ASSURES
n_courant   = N_ASSURES
age_courant = AGE_0
resultats_globaux = []

for sc in scenarios_annuels:
    res = inventaire_annuel(
        PM_courant, n_courant, age_courant,
        sc, hyp, df, RENTE_R, TAUX_I
    )

    print(f"\n{'─'*70}")
    print(f"  ANNÉE {sc['annee']} | Rendement réel : {sc['rendement_reel']*100:.1f}% "
          f"| Mortalité : {sc['tx_mortalite_reel']*100:.0f}% de l'attendue")
    print(f"{'─'*70}")
    print(f"  Assurés début     : {res['n_debut']:>8,}")
    print(f"  Décès attendus    : {res['deces_att']:>8.1f}  "
          f"| Décès réels     : {res['deces_reels']:>6.1f}")
    print(f"  Rachats           : {res['n_rachats']:>8.1f}  "
          f"| Assurés fin     : {res['n_fin']:>6,}")
    print()
    print(f"  Rentes versées    : {res['rentes_versees']:>12,.0f} €")
    print(f"  PM début          : {res['PM_deb']:>12,.0f} €")
    print(f"  PM fin            : {res['PM_fin']:>12,.0f} €")
    print()
    print(f"  {'Résultat financier':30} : {res['res_financier']:>+12,.0f} €"
          f"  (rendement {sc['rendement_reel']*100:.1f}% vs garanti {TAUX_I*100:.1f}%)")
    print(f"  {'Résultat mortalité':30} : {res['res_mortalite']:>+12,.0f} €"
          f"  (mortalité {sc['tx_mortalite_reel']*100:.0f}% de l'attendue)")
    print(f"  {'Résultat gestion':30} : {res['res_gestion']:>+12,.0f} €"
          f"  (frais {sc['frais_reels_pct']*100:.1f}% vs chargement {hyp['taux_gestion']*100:.1f}%)")
    print(f"  {'─'*54}")
    print(f"  {'RÉSULTAT TECHNIQUE TOTAL':30} : {res['res_total']:>+12,.0f} €")
    print(f"  {'Participation aux bénéfices':30} : {res['PB']:>12,.0f} €")
    print(f"  {'Pénalités de rachat':30} : {res['penalites']:>12,.0f} €")

    resultats_globaux.append({**sc, **res})
    PM_courant  = res["PM_fin"]
    n_courant   = res["n_fin"]
    age_courant += 1

df_res = pd.DataFrame(resultats_globaux)


# ─────────────────────────────────────────────────────────────
# CELLULE 5 : Graphique — décomposition du résultat
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

annees = df_res["annee"].values
x      = np.arange(len(annees))
width  = 0.25

# Graphique 1 : barres empilées résultat
bars_fin  = axes[0].bar(x - width, df_res["res_financier"],
                         width, label="Résultat financier", color="#3b82f6", alpha=0.85)
bars_mort = axes[0].bar(x,         df_res["res_mortalite"],
                         width, label="Résultat mortalité", color="#ef4444", alpha=0.85)
bars_gest = axes[0].bar(x + width, df_res["res_gestion"],
                         width, label="Résultat gestion",   color="#8b5cf6", alpha=0.85)

# Ligne résultat total
axes[0].plot(x, df_res["res_total"], "ko--", linewidth=2,
             markersize=7, label="Résultat total", zorder=5)
axes[0].axhline(y=0, color="black", linewidth=0.8, alpha=0.5)

axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Année {a}" for a in annees])
axes[0].set_ylabel("Résultat (€)")
axes[0].set_title("Décomposition du résultat technique")
axes[0].legend(fontsize=9)
axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:+,.0f}€"))
axes[0].grid(alpha=0.3, axis="y")

# Graphique 2 : évolution PM + rentes
axes[1].bar(x, df_res["rentes_versees"],
            color="#10b981", alpha=0.6, label="Rentes versées")
ax2b = axes[1].twinx()
pm_vals = [PM_0 * N_ASSURES] + list(df_res["PM_fin"])
ax2b.plot(range(-1, len(annees)), pm_vals,
          "s-", color="#f59e0b", linewidth=2,
          markersize=7, label="PM portefeuille")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"Année {a}" for a in annees])
axes[1].set_ylabel("Rentes versées (€)", color="#10b981")
ax2b.set_ylabel("PM portefeuille (€)", color="#f59e0b")
axes[1].tick_params(axis="y", labelcolor="#10b981")
ax2b.tick_params(axis="y", labelcolor="#f59e0b")
ax2b.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M€"))
axes[1].set_title("Rentes versées et évolution PM")
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("module4_inventaire_resultat.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────
# CELLULE 6 : La participation aux bénéfices (PB)
#
# COURS (Leçon 10) : La PB est la redistribution aux assurés
# d'une partie des bénéfices réalisés par l'assureur.
#
# Minimum légal (Code des assurances) :
#   - 85% du résultat financier
#   - 90% du résultat technique (mortalité + gestion)
#
# La PB peut être versée immédiatement ou mise en réserve
# (provision pour participation aux bénéfices = PPB)
# La PPB est une "réserve de lissage" — très utilisée en France.
# ─────────────────────────────────────────────────────────────

print("\nPARTICIPATION AUX BÉNÉFICES — Analyse\n")

for _, row in df_res.iterrows():
    RF  = row["res_financier"]
    RMG = row["res_mortalite"] + row["res_gestion"]  # hors financier

    PB_fin  = max(0, 0.85 * RF)
    PB_tech = max(0, 0.90 * RMG)
    PB_tot  = PB_fin + PB_tech

    print(f"Année {int(row['annee'])} :")
    print(f"  Résultat financier            : {RF:>+12,.0f} €")
    print(f"  PB financière (85% × RF)      : {PB_fin:>+12,.0f} €")
    print(f"  Résultat tech. (mort. + gest.): {RMG:>+12,.0f} €")
    print(f"  PB technique (90% × RT)       : {PB_tech:>+12,.0f} €")
    print(f"  ──────────────────────────────────────────")
    print(f"  PB TOTALE MINIMALE            : {PB_tot:>+12,.0f} €")

    n = row["n_debut"]
    if n > 0:
        print(f"  PB par assuré                 : {PB_tot/n:>+12,.2f} €")
    print()

print("→ En année 1 (bonne année), les assurés bénéficient d'une PB")
print("  grâce au rendement élevé des placements.")
print("→ En année 3 (mauvaise année), le résultat financier < 0 :")
print("  aucune PB légale obligatoire — mais l'assureur peut puiser")
print("  dans sa PPB (provision pour PB accumulée) pour lisser.")


# ─────────────────────────────────────────────────────────────
# CELLULE 7 : Gestion actif-passif — notion clé (Leçon 10)
#
# COURS : L'assureur doit aligner son actif (placements)
# et son passif (PM). Le risque principal :
#   - Passif : taux garanti fixe (ex 2%) sur 20-30 ans
#   - Actif  : rendements variables selon les marchés
#
# Duration : mesure la sensibilité de la PM aux taux
#   Duration PM ≈ ex = 18,71 ans pour une rente à 65 ans
#   L'actif doit avoir une duration similaire (immunisation)
# ─────────────────────────────────────────────────────────────

print("\nGESTION ACTIF-PASSIF — Duration et risque de taux\n")

def duration_PM(table, age_x, rente_R, taux_i):
    """
    Duration de Macaulay de la PM.
    Duration = Σ(t × flux_t) / Σ(flux_t)
    Mesure la sensibilité de la PM à une variation de taux.
    """
    v   = 1 / (1 + taux_i)
    lx  = table.loc[table["age"] == age_x, "lx"].values[0]
    num = 0.0   # numérateur Σ(t × flux)
    den = 0.0   # dénominateur Σ(flux) = PM

    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110: break
        St    = table.loc[table["age"] == age_t, "lx"].values[0] / lx
        flux  = rente_R * St * (v ** t)
        num  += t * flux
        den  += flux

    return num / den if den > 0 else 0

dur_65 = duration_PM(df, AGE_0, RENTE_R, TAUX_I)
dur_70 = duration_PM(df, 70,    RENTE_R, TAUX_I)
dur_75 = duration_PM(df, 75,    RENTE_R, TAUX_I)

print(f"Duration de la PM (i={TAUX_I*100}%) :\n")
print(f"  Rente depuis 65 ans : {dur_65:.2f} ans")
print(f"  Rente depuis 70 ans : {dur_70:.2f} ans")
print(f"  Rente depuis 75 ans : {dur_75:.2f} ans")
print()
print(f"Sensibilité au taux (duration modifiée) :")
dur_mod_65 = dur_65 / (1 + TAUX_I)
print(f"  Si taux monte de +1% → PM baisse de {dur_mod_65:.2f}%")
print(f"  Si taux baisse de -1% → PM monte de {dur_mod_65:.2f}%")
print()
print(f"Implication pour la gestion actif-passif :")
print(f"  L'actif doit avoir une duration ≈ {dur_65:.1f} ans")
print(f"  pour immuniser le portefeuille contre le risque de taux.")
print(f"  → On privilégie des obligations longues (OAT 20-30 ans)")
print(f"    pour matcher la duration du passif retraite.")


# ─────────────────────────────────────────────────────────────
# CELLULE 8 : Formulations entretien — Module 4 complet
# ─────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FORMULATIONS ENTRETIEN — Module 4 : Inventaire & Résultats         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Q : "Comment fonctionne l'inventaire trimestriel ?"                 ║
║  R : "L'inventaire reconstitue la PM de fin de période à partir      ║
║       de la PM de début, en intégrant les flux réels : rentes        ║
║       versées, décès, rachats et produits financiers. On compare     ║
║       chaque poste aux hypothèses techniques pour décomposer         ║
║       le résultat en trois composantes."                             ║
║                                                                      ║
║  Q : "C'est quoi le résultat mortalité sur une rente ?"              ║
║  R : "Pour une rente, moins de décès que prévu = perte, car          ║
║       l'assureur paie plus longtemps. C'est le risque de             ║
║       longévité. Si la mortalité réelle est à 75% de la              ║
║       table, le résultat mortalité est négatif."                     ║
║                                                                      ║
║  Q : "C'est quoi la participation aux bénéfices ?"                   ║
║  R : "La PB est la redistribution aux assurés d'une partie           ║
║       des bénéfices : 85% du résultat financier et 90% du           ║
║       résultat technique au minimum. L'excédent peut aller           ║
║       en PPB pour lisser les années difficiles."                     ║
║                                                                      ║
║  Q : "C'est quoi la duration et pourquoi c'est important ?"          ║
║  R : "La duration mesure la sensibilité de la PM aux taux.           ║
║       Pour une rente à 65 ans, elle est d'environ 12 ans.            ║
║       L'actif doit avoir une duration similaire pour immuniser        ║
║       le portefeuille — c'est la base de l'ALM."                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("🎉 MODULE 4 COMPLET :")
print("   ✅ Inventaire trimestriel — équation de passage PM")
print("   ✅ Décomposition résultat : financier + mortalité + gestion")
print("   ✅ Participation aux bénéfices (PB et PPB)")
print("   ✅ Duration et gestion actif-passif (ALM)")
print()
print("→ DERNIER MODULE : Module 5 — Backtesting PM")
print("  (Analyse N/N+1, écarts, revue des hypothèses)")
