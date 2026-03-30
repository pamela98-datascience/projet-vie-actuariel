# ============================================================
# MODULE 3 — PROVISIONS MATHÉMATIQUES (Leçons 6-7)
# Notions 9, 10, 11, 12
# Projet Vie Actuariel — Pamela Fagla (ISUP M1 Actuariat)
# Google Colab — à exécuter cellule par cellule
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────
# CELLULE 1 : Rechargement table + fonctions Modules 1 & 2
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

def annuite_actuarielle(table, age_x, taux_i, rente_R=1.0):
    """äx = Σ S(t) × vᵗ — valeur d'une rente de 1€/an depuis age_x"""
    v   = 1 / (1 + taux_i)
    lx  = table.loc[table["age"] == age_x, "lx"].values[0]
    a_x = 0.0
    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110: break
        St  = table.loc[table["age"] == age_t, "lx"].values[0] / lx
        a_x += St * (v ** t)
    return a_x * rente_R

print("✅ Table + fonctions rechargées — prêt pour Module 3")


# ═════════════════════════════════════════════════════════════
# NOTION 9 : PM prospective sur rente viagère
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 2 : Calcul de la PM à la souscription (t=0)
#
# COURS : Pour une rente viagère pure (pas de primes futures),
#         la PM à t=0 = VA(toutes les rentes futures)
#         PM₀ = R × äx   (ce qu'on a calculé au Module 2)
#
# C'est la provision que l'assureur doit constituer
# le jour où le retraité commence à percevoir sa rente.
# ─────────────────────────────────────────────────────────────

# Paramètres du contrat de référence
age_souscription = 65
rente_R          = 14_400   # 1 200 €/mois = 14 400 €/an
taux_i           = 0.02
v                = 1 / (1 + taux_i)

# PM à la souscription
PM_0 = annuite_actuarielle(df, age_souscription, taux_i, rente_R)

print(f"NOTION 9 : PM prospective — Rente {rente_R:,}€/an | Age {age_souscription} | i={taux_i*100}%\n")
print(f"PM à la souscription (t=0) : {PM_0:,.2f} €")
print(f"Vérification : äx = PM / R = {PM_0/rente_R:.4f} années")
print(f"(cohérent avec e{age_souscription} = {df.loc[df['age']==age_souscription,'ex'].values[0]:.2f} ans — proche mais pas égal car on actualise)")


# ─────────────────────────────────────────────────────────────
# CELLULE 3 : Calcul de la PM flux par flux (détail pédagogique)
#
# COURS : On décompose la PM en flux annuels pour comprendre
#         la structure. Chaque flux = R × S(t) × vᵗ
#         La PM = somme de tous ces flux.
# ─────────────────────────────────────────────────────────────

lx_65 = df.loc[df["age"] == age_souscription, "lx"].values[0]

flux_data = []
PM_cumul = 0.0

for t in range(111 - age_souscription):
    age_t = age_souscription + t
    if age_t > 110: break

    lxt = df.loc[df["age"] == age_t, "lx"].values[0]
    St  = lxt / lx_65
    vt  = v ** t
    flux = rente_R * St * vt
    PM_cumul += flux

    flux_data.append({
        "t": t, "age": age_t,
        "S(t)": round(St, 5),
        "vᵗ": round(vt, 5),
        "S(t)×vᵗ": round(St * vt, 5),
        "Flux R×S(t)×vᵗ": round(flux, 2),
        "PM cumulée": round(PM_cumul, 2),
    })

df_flux = pd.DataFrame(flux_data)

print("\nDétail des 15 premiers flux :")
print(df_flux.head(15).to_string(index=False))
print(f"\n... Total : {len(df_flux)} flux")
print(f"PM totale = {df_flux['Flux R×S(t)×vᵗ'].sum():,.2f} €")

# À quel horizon a-t-on capitalisé 50% de la PM ?
seuil_50 = PM_0 * 0.5
t_50 = df_flux[df_flux["PM cumulée"] >= seuil_50]["t"].iloc[0]
print(f"\n50% de la PM est constituée par les flux des {t_50} premières années")
print(f"(âge {age_souscription + t_50} ans) — utile pour la gestion actif-passif")


# ═════════════════════════════════════════════════════════════
# NOTION 10 : Évolution de la PM dans le temps
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 4 : La PM évolue chaque année — pourquoi ?
#
# COURS : La PM est recalculée à chaque fin d'exercice.
#         Elle évolue sous l'effet de 3 facteurs :
#
#   1. RENTES VERSÉES : on a payé R dans l'année → PM diminue
#   2. INTÉRÊTS : les placements ont rapporté i×PM → PM augmente
#   3. MORTALITÉ : des assurés sont décédés → PM diminue
#
# Équation de récurrence (loi de Thiele simplifiée) :
#   PM(t+1) = [PM(t) × (1+i) - R] × px+t
#             ↑ capitalisation    ↑ survie
# ─────────────────────────────────────────────────────────────

def evolution_PM(df, age_depart, rente_R, taux_i, n_annees=45):
    """
    Calcule l'évolution de la PM année par année.
    Deux méthodes — doivent donner le même résultat :
      1. Recalcul prospectif à chaque date (méthode exacte)
      2. Récurrence de Thiele (méthode de vérification)
    """
    v     = 1 / (1 + taux_i)
    resultats = []

    for t in range(n_annees + 1):
        age_t = age_depart + t
        if age_t > 105: break

        # Méthode 1 : recalcul prospectif exact
        PM_prosp = annuite_actuarielle(df, age_t, taux_i, rente_R)

        # Nombre de survivants à cet âge (pour info)
        lxt = df.loc[df["age"] == age_t, "lx"].values[0]
        St  = lxt / df.loc[df["age"] == age_depart, "lx"].values[0]

        resultats.append({
            "t": t,
            "age": age_t,
            "S(t)": round(St, 5),
            "PM prospective": round(PM_prosp, 2),
            "Rente annuelle": rente_R,
        })

    return pd.DataFrame(resultats)

df_evol = evolution_PM(df, age_souscription, rente_R, taux_i)

print("\nNOTION 10 : Évolution de la PM dans le temps\n")
print(f"{'t':>4} | {'Âge':>5} | {'S(t)':>8} | {'PM (€)':>14} | {'PM / PM₀':>10}")
print("-" * 55)
for _, r in df_evol[df_evol["t"].isin([0,5,10,15,20,25,30,35,40])].iterrows():
    ratio = r["PM prospective"] / PM_0
    print(f"{int(r['t']):>4} | {int(r['age']):>5} | {r['S(t)']:>8.4f} | "
          f"{r['PM prospective']:>14,.2f} | {ratio:>9.2%}")

# Vérification par récurrence de Thiele
print("\nVérification — Récurrence de Thiele :")
print("PM(t+1) = [PM(t) × (1+i) − R] × px+t\n")
PM_thiele = PM_0
for t in range(5):
    age_t  = age_souscription + t
    px_t   = df.loc[df["age"] == age_t, "px"].values[0]
    PM_new = (PM_thiele * (1 + taux_i) - rente_R) * px_t
    PM_ref = df_evol.loc[df_evol["t"] == t+1, "PM prospective"].values[0]
    print(f"  t={t}: PM({t+1}) Thiele={PM_new:,.2f}  |  Prospectif={PM_ref:,.2f}  |  "
          f"Écart={abs(PM_new-PM_ref):.2f} €")
    PM_thiele = PM_new


# ─────────────────────────────────────────────────────────────
# CELLULE 5 : Graphique évolution PM
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : PM en valeur absolue
axes[0].plot(df_evol["age"], df_evol["PM prospective"],
             color="#8b5cf6", linewidth=2.5)
axes[0].fill_between(df_evol["age"], df_evol["PM prospective"],
                     alpha=0.1, color="#8b5cf6")
axes[0].axhline(y=PM_0, color="gray", linestyle="--",
                alpha=0.5, label=f"PM₀ = {PM_0:,.0f}€")
axes[0].set_xlabel("Âge de l'assuré")
axes[0].set_ylabel("PM (€)")
axes[0].set_title("Évolution de la PM — Rente viagère 65 ans")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}€"))
axes[0].legend()
axes[0].grid(alpha=0.3)

# Graphique 2 : PM / PM₀ + S(t) superposés
ax2b = axes[1].twinx()
axes[1].plot(df_evol["age"], df_evol["PM prospective"] / PM_0 * 100,
             color="#8b5cf6", linewidth=2.5, label="PM / PM₀ (%)")
ax2b.plot(df_evol["age"], df_evol["S(t)"] * 100,
          color="#10b981", linewidth=2, linestyle="--", label="S(t) (%)")
axes[1].set_xlabel("Âge")
axes[1].set_ylabel("PM / PM₀ (%)", color="#8b5cf6")
ax2b.set_ylabel("S(t) %", color="#10b981")
axes[1].set_title("PM normalisée vs Loi de survie S(t)")
axes[1].tick_params(axis="y", labelcolor="#8b5cf6")
ax2b.tick_params(axis="y", labelcolor="#10b981")
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("module3_evolution_PM.png", dpi=150, bbox_inches="tight")
plt.show()

print("Observation :")
print("→ La PM décroît plus vite que S(t)")
print("  Car en plus de la mortalité, les rentes versées 'consomment' la PM")
print("  et le temps qui passe réduit l'horizon des flux futurs.")


# ═════════════════════════════════════════════════════════════
# NOTION 11 : Rachat et valeur de rachat
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 6 : Le rachat — quand l'assuré veut sortir
#
# COURS (Leçon 7) : L'assuré peut demander à "racheter" son
# contrat avant le terme — récupérer son épargne.
#
# Valeur de rachat = PM × (1 − taux de rachat)
#   taux_rachat = pénalité fixée au contrat (ex : 2-5%)
#   Elle compense le coût de sortie anticipée pour l'assureur.
#
# RACHAT TOTAL   : l'assuré sort complètement
# RACHAT PARTIEL : l'assuré retire une fraction de la PM
# RÉDUCTION      : l'assuré arrête de payer mais garde le contrat
#                  → la prestation future est réduite
# ─────────────────────────────────────────────────────────────

print("\nNOTION 11 : Valeur de rachat\n")

taux_rachat = 0.02   # pénalité 2%

print(f"Taux de pénalité rachat : {taux_rachat*100}%\n")
print(f"{'t':>4} | {'Âge':>5} | {'PM (€)':>14} | {'Valeur rachat (€)':>18} | {'Pénalité (€)':>13}")
print("-" * 65)

ages_rachat = [0, 5, 10, 15, 20]
for t in ages_rachat:
    row = df_evol[df_evol["t"] == t].iloc[0]
    PM_t     = row["PM prospective"]
    VR_t     = PM_t * (1 - taux_rachat)
    penalite = PM_t * taux_rachat
    print(f"{t:>4} | {int(row['age']):>5} | {PM_t:>14,.2f} | {VR_t:>18,.2f} | {penalite:>13,.2f}")

print(f"\nInterprétation t=10 (âge 75 ans) :")
PM_10 = df_evol.loc[df_evol["t"]==10, "PM prospective"].values[0]
VR_10 = PM_10 * (1 - taux_rachat)
print(f"  PM = {PM_10:,.2f}€ → l'assuré récupère {VR_10:,.2f}€")
print(f"  L'assureur garde {PM_10*taux_rachat:,.2f}€ de pénalité")
print(f"  (pour couvrir ses coûts de liquidation anticipée)")

# Rachat partiel : retirer 20% de la PM
print(f"\nRachat partiel de 20% à t=10 :")
pct_rachat = 0.20
montant_rachete = PM_10 * pct_rachat * (1 - taux_rachat)
PM_apres = PM_10 * (1 - pct_rachat)
rente_reduite = rente_R * (1 - pct_rachat)
print(f"  Montant versé à l'assuré  : {montant_rachete:,.2f}€")
print(f"  PM résiduelle             : {PM_apres:,.2f}€")
print(f"  Rente future réduite à    : {rente_reduite:,.2f}€/an ({rente_reduite/12:,.2f}€/mois)")


# ═════════════════════════════════════════════════════════════
# NOTION 12 : Best Estimate S2 / IFRS17
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 7 : Best Estimate — la PM vue par Solvabilité 2
#
# COURS (Leçon 7) : Sous Solvabilité 2, on ne calcule plus la
# PM avec un taux garanti fixe. On utilise :
#
# BE = Σ flux_futurs × facteur_survie × facteur_actualisation_marché
#
# Différences clés PM classique vs Best Estimate :
#   - Taux d'actualisation : taux technique fixe → courbe des taux EIOPA
#   - Hypothèses mortalité : tables réglementaires → tables d'expérience
#   - Inclut tous les cash-flows : rentes, rachats, frais de gestion
#
# Ici on simule avec une courbe des taux simplifiée
# (Nelson-Siegel sera dans la page Streamlit)
# ─────────────────────────────────────────────────────────────

print("\nNOTION 12 : Best Estimate Solvabilité 2\n")

def courbe_taux_simplifiee(t, beta0=0.03, beta1=-0.02, tau=5.0):
    """
    Courbe de taux zéro-coupon simplifiée (Nelson-Siegel).
    beta0  = taux long terme (niveau)
    beta1  = spread court/long (pente)
    tau    = paramètre de forme
    Valeurs typiques zone euro 2024 : beta0 ≈ 3%, taux courts ≈ 1-2%
    """
    if t == 0:
        return beta0 + beta1
    facteur = (1 - np.exp(-t/tau)) / (t/tau)
    return beta0 + beta1 * facteur

# Comparaison BE selon différentes courbes de taux
scenarios_taux = {
    "Taux fixe 2% (PM classique)":    lambda t: 0.02,
    "Courbe plate 1%":                 lambda t: 0.01,
    "Courbe plate 3%":                 lambda t: 0.03,
    "Nelson-Siegel (β0=3%, β1=-2%)":  lambda t: courbe_taux_simplifiee(t),
}

def calcul_BE(df, age_x, rente_R, courbe_fn):
    """Calcule le Best Estimate avec une courbe de taux quelconque."""
    lx  = df.loc[df["age"] == age_x, "lx"].values[0]
    BE  = 0.0
    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110: break
        St  = df.loc[df["age"] == age_t, "lx"].values[0] / lx
        i_t = courbe_fn(t)
        vt  = 1 / (1 + i_t) ** t if t > 0 else 1.0
        BE += rente_R * St * vt
    return BE

print(f"Comparaison BE selon la courbe des taux")
print(f"Contrat : Rente {rente_R:,}€/an | Homme {age_souscription} ans\n")
print(f"{'Scénario':45} | {'BE (€)':>14} | {'vs PM classique':>16}")
print("-" * 80)

PM_ref = calcul_BE(df, age_souscription, rente_R, scenarios_taux["Taux fixe 2% (PM classique)"])

for nom, fn in scenarios_taux.items():
    BE = calcul_BE(df, age_souscription, rente_R, fn)
    ecart = BE - PM_ref
    signe = "+" if ecart >= 0 else ""
    print(f"  {nom:43} | {BE:>14,.2f} | {signe}{ecart:>14,.2f}")

print(f"\nConclusion :")
print(f"→ Un taux plus bas → BE plus élevé → plus de provisions à constituer")
print(f"→ C'est le risque de taux : si les taux baissent, le BE monte")
print(f"  et l'assureur doit renforcer ses fonds propres (SCR taux sous S2)")

# SCR taux simplifié (choc ±100 bps sur la courbe plate)
print(f"\nSCR taux (choc réglementaire S2 ±100bps) :")
BE_base  = calcul_BE(df, age_souscription, rente_R, lambda t: 0.02)
BE_down  = calcul_BE(df, age_souscription, rente_R, lambda t: 0.01)  # -100bps
BE_up    = calcul_BE(df, age_souscription, rente_R, lambda t: 0.03)  # +100bps

SCR_down = BE_down - BE_base
SCR_up   = BE_base - BE_up

print(f"  BE base (i=2%)          : {BE_base:>14,.2f} €")
print(f"  BE choc down (i=1%)     : {BE_down:>14,.2f} € → hausse provisions : +{SCR_down:,.2f}€")
print(f"  BE choc up   (i=3%)     : {BE_up:>14,.2f} € → baisse provisions  : -{abs(SCR_up):,.2f}€")
print(f"  SCR taux (pire des deux): {max(SCR_down, abs(SCR_up)):>14,.2f} €")


# ─────────────────────────────────────────────────────────────
# CELLULE 8 : Graphique comparatif PM classique vs BE
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

configs = [
    (lambda t: 0.01, "#ef4444", "i=1% (BE prudent)"),
    (lambda t: 0.02, "#8b5cf6", "i=2% (PM classique)"),
    (lambda t: 0.03, "#10b981", "i=3% (BE optimiste)"),
    (lambda t: courbe_taux_simplifiee(t), "#f59e0b", "Nelson-Siegel"),
]

for fn, couleur, label in configs:
    vals = []
    for t in range(41):
        age_t = age_souscription + t
        if age_t > 105: break
        BE_t = calcul_BE(df, age_t, rente_R, fn)
        vals.append((age_t, BE_t))
    ages_v, bes_v = zip(*vals)
    ax.plot(ages_v, bes_v, color=couleur, linewidth=2, label=label)

ax.set_xlabel("Âge de l'assuré")
ax.set_ylabel("PM / Best Estimate (€)")
ax.set_title("Évolution PM classique vs Best Estimate S2 — Rente 14 400€/an")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}€"))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("module3_PM_vs_BE.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────
# CELLULE 9 : Formulations entretien — Module 3 complet
# ─────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FORMULATIONS ENTRETIEN — Module 3 : Provisions Mathématiques       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Q : "C'est quoi une provision mathématique ?"                       ║
║  R : "La PM est la valeur actuelle des engagements futurs de         ║
║       l'assureur envers ses assurés, nette des primes futures.       ║
║       Pour une rente viagère, PM = R × äx+t, recalculée             ║
║       à chaque date t. C'est le poste principal du passif."          ║
║                                                                      ║
║  Q : "Comment la PM évolue-t-elle dans le temps ?"                   ║
║  R : "La PM diminue sous 3 effets : les rentes versées la            ║
║       consomment, la mortalité fait sortir des assurés, et           ║
║       l'horizon des flux futurs se réduit. Elle est vérifiée         ║
║       par la récurrence de Thiele : PM(t+1) = [PM(t)×(1+i)−R]×px." ║
║                                                                      ║
║  Q : "Quelle est la différence PM classique et Best Estimate ?"      ║
║  R : "La PM classique utilise un taux technique fixe réglementaire.  ║
║       Le Best Estimate S2 utilise la courbe des taux EIOPA sans      ║
║       marge de prudence. Il est plus sensible aux variations de      ║
║       taux — c'est ce qui génère le SCR taux."                       ║
║                                                                      ║
║  Q : "C'est quoi un rachat ?"                                        ║
║  R : "Un rachat c'est la sortie anticipée d'un assuré. Il reçoit     ║
║       la valeur de rachat = PM × (1 − taux pénalité). L'assureur    ║
║       gère le risque de rachat car un pic de rachats peut            ║
║       créer un besoin de liquidité brutal."                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("🎉 MODULE 3 COMPLET — 4 notions maîtrisées :")
print("   ✅ Notion 9  : PM prospective rente viagère")
print("   ✅ Notion 10 : Évolution PM dans le temps (récurrence Thiele)")
print("   ✅ Notion 11 : Rachat total, partiel, réduction")
print("   ✅ Notion 12 : Best Estimate S2 et SCR taux")
print()
print("→ Prochain module : Module 4 — Inventaire & résultats")
print("  (compte technique, résultat mortalité, PB — Leçons 8-10)")
