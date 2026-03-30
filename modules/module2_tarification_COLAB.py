# ============================================================
# MODULE 2 — TARIFICATION (Leçons 4-5)
# Notions 5, 6, 7, 8
# Projet Vie Actuariel — Pamela Fagla (ISUP M1 Actuariat)
# Google Colab — à exécuter cellule par cellule
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CELLULE 1 : Rechargement table complète (Module 1)
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

print("✅ Table Module 1 rechargée")


# ═════════════════════════════════════════════════════════════
# NOTION 5 : Le taux d'intérêt technique et vᵗ
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 2 : vᵗ — le facteur d'actualisation
#
# COURS : vᵗ = 1 / (1+i)ᵗ
#         i = taux technique (rendement garanti par l'assureur)
#         Plus i est élevé, plus vᵗ est petit
#         → la prime actuelle est plus faible
#
# Réglementation française : le taux technique est plafonné
# par l'ACPR pour éviter que l'assureur sur-promette.
# ─────────────────────────────────────────────────────────────

# Trois taux pour comparer l'effet
taux = {"i=0%": 0.00, "i=1%": 0.01, "i=2%": 0.02, "i=3%": 0.03}

horizons = np.arange(0, 46)  # t de 0 à 45 ans

print("NOTION 5 : Facteurs d'actualisation vᵗ\n")
print(f"{'t':>4}", end="")
for nom in taux:
    print(f"  {nom:>10}", end="")
print()
print("-" * 50)

for t in [0, 5, 10, 20, 30, 40]:
    print(f"{t:>4}", end="")
    for i in taux.values():
        vt = 1 / (1 + i) ** t
        print(f"  {vt:>10.6f}", end="")
    print()

print("\n→ Avec i=3%, 1€ dans 20 ans ne vaut que 0.554€ aujourd'hui.")
print("  L'assureur doit donc provisionner moins grâce au rendement garanti.")

# Graphique : vᵗ selon le taux
fig, ax = plt.subplots(figsize=(10, 5))
couleurs_taux = ["#94a3b8", "#10b981", "#f59e0b", "#ef4444"]

for (nom, i), couleur in zip(taux.items(), couleurs_taux):
    vt_vals = 1 / (1 + i) ** horizons
    ax.plot(horizons, vt_vals, linewidth=2, label=nom, color=couleur)

ax.set_xlabel("Horizon t (années)")
ax.set_ylabel("vᵗ = facteur d'actualisation")
ax.set_title("Effet du taux technique sur l'actualisation")
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="50%")
plt.tight_layout()
plt.savefig("vt_actualisation.png", dpi=150, bbox_inches="tight")
plt.show()


# ═════════════════════════════════════════════════════════════
# NOTION 6 : Prime pure — assurance décès temporaire
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 3 : Formule prime pure décès
#
# COURS : Contrat décès temporaire n ans :
#   - Si l'assuré décède dans les n ans → l'assureur verse C
#   - Si il survit → rien
#
# Prime pure = espérance du flux actualisé
#   PP = C × Σ(t=0 à n-1) [ tpx × qx+t × v^(t+1) ]
#
# Décomposition :
#   tpx       = proba d'être en vie à t (= S(t) depuis x)
#   qx+t      = proba de décéder entre t et t+1
#   tpx×qx+t  = proba de décéder exactement à t
#   v^(t+1)   = on actualise au moment du décès (fin d'année)
# ─────────────────────────────────────────────────────────────

def prime_pure_deces(df, age_x, n_ans, capital_C, taux_i):
    """
    Calcule la prime pure unique d'une assurance décès temporaire n ans.

    Paramètres :
      age_x    : âge de l'assuré à la souscription
      n_ans    : durée de la garantie (en années)
      capital_C: capital versé en cas de décès
      taux_i   : taux technique annuel

    Retourne :
      prime pure unique (montant à payer en une fois aujourd'hui)
      + DataFrame des flux annuels pour analyse
    """
    v = 1 / (1 + taux_i)

    lx = df.loc[df["age"] == age_x, "lx"].values[0]

    flux_list = []
    pp = 0.0

    for t in range(n_ans):
        age_t  = age_x + t
        age_t1 = age_x + t + 1

        if age_t1 > 110:
            break

        lxt  = df.loc[df["age"] == age_t,  "lx"].values[0]
        lxt1 = df.loc[df["age"] == age_t1, "lx"].values[0]

        # tpx = probabilité d'être en vie à t depuis x
        tpx = lxt / lx

        # qx+t = probabilité de décéder entre t et t+1
        qxt = df.loc[df["age"] == age_t, "qx"].values[0]

        # Probabilité de décéder exactement entre t et t+1
        prob_deces_t = tpx * qxt

        # Facteur d'actualisation (paiement en fin d'année t+1)
        vt1 = v ** (t + 1)

        # Flux actualisé
        flux = capital_C * prob_deces_t * vt1

        pp += flux

        flux_list.append({
            "t": t,
            "age": age_t,
            "tpx": round(tpx, 6),
            "qx+t (‰)": round(qxt * 1000, 4),
            "prob_décès_t": round(prob_deces_t, 6),
            "v^(t+1)": round(vt1, 6),
            "flux (€)": round(flux, 2),
        })

    return pp, pd.DataFrame(flux_list)


# Application
age_assuré = 40
duree      = 10
capital    = 100_000
taux_i     = 0.02

pp_deces, df_flux_deces = prime_pure_deces(df, age_assuré, duree, capital, taux_i)

print(f"\nNOTION 6 : Prime pure — Assurance décès temporaire {duree} ans\n")
print(f"  Assuré : {age_assuré} ans | Capital : {capital:,} € | Taux : {taux_i*100}%\n")
print(df_flux_deces.to_string(index=False))
print(f"\n  Prime pure UNIQUE = {pp_deces:,.2f} €")
print(f"  Interprétation : l'assuré paie {pp_deces:,.2f} € aujourd'hui")
print(f"  pour être couvert {duree} ans avec un capital de {capital:,} €.")

# Sensibilité au taux
print(f"\nSensibilité de la prime pure au taux technique :")
for ti in [0.00, 0.01, 0.02, 0.03]:
    pp, _ = prime_pure_deces(df, age_assuré, duree, capital, ti)
    print(f"  i={ti*100:.0f}% → PP = {pp:,.2f} €")
print("→ Plus le taux est élevé, plus la prime est faible (l'assureur rentabilise mieux ses placements)")


# ═════════════════════════════════════════════════════════════
# NOTION 7 : Prime pure — rente viagère
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 4 : Formule rente viagère (annuité actuarielle)
#
# COURS : Contrat rente viagère :
#   L'assureur verse R€/an tant que l'assuré est en vie.
#
# Prime pure unique = äx (annuité actuarielle de vie)
#   PP = R × äx = R × Σ(t=0 à omega-x) [ S(t) × vᵗ ]
#
# C'est la somme des flux futurs pondérés par :
#   - S(t) : probabilité d'être en vie pour le recevoir
#   - vᵗ   : actualisation de ce flux
#
# Note : äx (avec tréma) = rente versée en DEBUT d'année (immédiate)
#        ax  (sans tréma) = rente versée en FIN d'année
# ─────────────────────────────────────────────────────────────

def annuite_actuarielle(df, age_x, taux_i, rente_R=1.0, debut_annee=True):
    """
    Calcule l'annuité actuarielle äx (ou ax).

    äx = Σ(t=0 à omega-x) S(t) × vᵗ   [début d'année]
    ax = Σ(t=1 à omega-x) S(t) × vᵗ   [fin d'année]

    Retourne :
      äx (ou ax) : valeur de l'annuité pour 1€/an
      prime pure pour rente_R €/an
      DataFrame des flux
    """
    v   = 1 / (1 + taux_i)
    lx  = df.loc[df["age"] == age_x, "lx"].values[0]

    flux_list = []
    a_x = 0.0

    t_depart = 0 if debut_annee else 1

    for t in range(t_depart, 111 - age_x):
        age_t = age_x + t
        if age_t > 110:
            break

        lxt = df.loc[df["age"] == age_t, "lx"].values[0]
        St  = lxt / lx          # S(t) = survie depuis x
        vt  = v ** t            # actualisation

        flux = rente_R * St * vt
        a_x += flux

        flux_list.append({
            "t": t,
            "age": age_t,
            "S(t)": round(St, 6),
            "vᵗ": round(vt, 6),
            "S(t)×vᵗ": round(St * vt, 6),
            "flux (€/an)": round(flux, 2),
        })

    return a_x, pd.DataFrame(flux_list)


# Application : rente viagère 1 200 €/mois = 14 400 €/an depuis 65 ans
age_retraite = 65
rente_annuelle = 14_400
taux_i = 0.02

a_65, df_flux_rente = annuite_actuarielle(df, age_retraite, taux_i, rente_annuelle)

print(f"\nNOTION 7 : Prime pure — Rente viagère depuis {age_retraite} ans\n")
print(f"  Rente : {rente_annuelle:,} €/an ({rente_annuelle//12:,} €/mois) | Taux : {taux_i*100}%\n")

# Affichage des 10 premières et dernières lignes
print("10 premiers flux :")
print(df_flux_rente.head(10).to_string(index=False))
print(f"\n... ({len(df_flux_rente)} années au total) ...\n")
print("5 derniers flux :")
print(df_flux_rente.tail(5).to_string(index=False))

print(f"\n  Prime pure UNIQUE (äx) = {a_65:,.2f} €")
print(f"  Interprétation : pour garantir {rente_annuelle:,}€/an à vie depuis {age_retraite} ans,")
print(f"  l'assureur doit constituer {a_65:,.2f} € de provision.")

# Comparaison sans actualisation
a_65_0, _ = annuite_actuarielle(df, age_retraite, 0.00, rente_annuelle)
print(f"\n  Sans actualisation (i=0%) : {a_65_0:,.2f} €")
print(f"  Avec i=2%                 : {a_65:,.2f} €")
print(f"  Économie grâce aux taux   : {a_65_0 - a_65:,.2f} €")

# Comparaison H/F
qx_tf = np.array([
    0.003052,0.000217,0.000154,0.000121,0.000100,0.000086,0.000075,
    0.000067,0.000062,0.000058,0.000057,0.000060,0.000072,0.000095,
    0.000126,0.000161,0.000196,0.000227,0.000250,0.000264,0.000270,
    0.000272,0.000274,0.000277,0.000281,0.000287,0.000294,0.000303,
    0.000313,0.000325,0.000339,0.000356,0.000376,0.000400,0.000428,
    0.000461,0.000499,0.000543,0.000593,0.000649,0.000712,0.000783,
    0.000863,0.000951,0.001049,0.001157,0.001276,0.001407,0.001551,
    0.001709,0.001882,0.002072,0.002280,0.002508,0.002758,0.003031,
    0.003330,0.003657,0.004014,0.004403,0.004827,0.005289,0.005792,
    0.006340,0.006937,0.007586,0.008292,0.009060,0.009893,0.010797,
    0.011778,0.012841,0.013992,0.015238,0.016586,0.018043,0.019617,
    0.021315,0.023145,0.025117,0.027237,0.029516,0.031962,0.034584,
    0.037393,0.040397,0.043606,0.047030,0.050681,0.054570,0.058705,
    0.063096,0.067753,0.072688,0.077909,0.083425,0.089244,0.095377,
    0.101832,0.108615,0.115734,0.123194,0.130999,0.139156,0.147668,
    0.156538,0.165766,0.175353,0.185300,0.195604,1.000000
])
df_f = pd.DataFrame({"age": ages, "qx": qx_tf})
df_f["px"]      = 1 - df_f["qx"]
df_f["lx"]      = l0 * np.concatenate([[1.0], np.cumprod(df_f["px"].values[:-1])])

a_65_f, _ = annuite_actuarielle(df_f, age_retraite, taux_i, rente_annuelle)
print(f"\n  Comparaison H/F (même rente {rente_annuelle:,}€/an, i={taux_i*100}%) :")
print(f"  Homme (TH) : {a_65:,.2f} €")
print(f"  Femme (TF) : {a_65_f:,.2f} €")
print(f"  Surcoût F  : {a_65_f - a_65:,.2f} €")


# ═════════════════════════════════════════════════════════════
# NOTION 8 : Chargements et prime nette
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CELLULE 5 : Les chargements
#
# COURS : La prime pure couvre le risque.
# Mais l'assureur a aussi des coûts :
#   α  = taux de chargement en acquisition (commission commercial)
#   β  = taux de chargement en gestion (frais annuels)
#
# Prime nette unique   = PP / (1 - α)
# Prime nette annuelle = PP + β × äx  (si frais annuels)
#
# En pratique, les chargements sont fixés par la direction
# commerciale et validés par l'actuaire tarifaire.
# ─────────────────────────────────────────────────────────────

print("\nNOTION 8 : Chargements et prime nette\n")

# Paramètres typiques du marché
alpha = 0.05   # 5% chargement acquisition (commission)
beta  = 0.008  # 0.8% chargement gestion annuel

# Exemple : rente 14 400 €/an depuis 65 ans
pp_unique = a_65   # prime pure unique calculée en Notion 7

# Prime nette unique (chargement acquisition)
prime_nette_unique = pp_unique / (1 - alpha)

# Si on veut convertir en prime annuelle sur n années
# On utilise äx pour la conversion unique → annuelle
# Prime annuelle P telle que P × äx = prime_nette_unique
ax_65, _ = annuite_actuarielle(df, age_retraite, taux_i, rente_R=1.0)
prime_annuelle = prime_nette_unique / ax_65

print(f"Contrat : Rente {rente_annuelle:,}€/an | Homme {age_retraite} ans | i={taux_i*100}%\n")
print(f"  Prime pure unique          : {pp_unique:>12,.2f} €")
print(f"  Chargement acquisition (α) : {alpha*100}%")
print(f"  Prime nette unique         : {prime_nette_unique:>12,.2f} €")
print(f"  Prime nette annuelle       : {prime_annuelle:>12,.2f} €/an")
print(f"                             = {prime_annuelle/12:>12,.2f} €/mois")
print()

# Décomposition de la prime nette
print("  Décomposition de la prime nette :")
print(f"    Coût du risque (PP)    : {pp_unique:>12,.2f} € ({pp_unique/prime_nette_unique*100:.1f}%)")
print(f"    Chargement acquisition : {prime_nette_unique - pp_unique:>12,.2f} € ({(prime_nette_unique-pp_unique)/prime_nette_unique*100:.1f}%)")

# Sensibilité aux chargements
print(f"\n  Sensibilité aux chargements (PP = {pp_unique:,.0f}€) :")
print(f"  {'α':>6} | {'Prime nette':>14} | {'Surcoût / PP':>14}")
print("  " + "-" * 40)
for a in [0.00, 0.03, 0.05, 0.08, 0.10]:
    pn = pp_unique / (1 - a)
    print(f"  {a*100:>5.0f}% | {pn:>14,.2f} € | {pn - pp_unique:>14,.2f} €")


# ─────────────────────────────────────────────────────────────
# CELLULE 6 : Graphique récapitulatif — décomposition prime
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : flux actualisés S(t)×vᵗ
df_flux_rente["S_t_vt"] = df_flux_rente["S(t)"] * df_flux_rente["vᵗ"]
axes[0].bar(df_flux_rente["age"], df_flux_rente["flux (€/an)"],
            color="#8b5cf6", alpha=0.6)
axes[0].plot(df_flux_rente["age"], df_flux_rente["flux (€/an)"],
             color="#6d28d9", linewidth=1.5)
axes[0].set_xlabel("Âge")
axes[0].set_ylabel("Flux annuel pondéré (€)")
axes[0].set_title(f"Flux rente × S(t) × vᵗ depuis {age_retraite} ans (i={taux_i*100}%)")
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
axes[0].grid(alpha=0.3)

# Graphique 2 : Camembert décomposition prime nette
labels   = ["Coût risque (PP)", f"Chargement α={alpha*100:.0f}%"]
values   = [pp_unique, prime_nette_unique - pp_unique]
couleurs = ["#10b981", "#f59e0b"]
axes[1].pie(values, labels=labels, colors=couleurs, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 11})
axes[1].set_title("Décomposition prime nette")

plt.tight_layout()
plt.savefig("module2_tarification.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────
# CELLULE 7 : Formulations entretien — Module 2 complet
# ─────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  FORMULATIONS ENTRETIEN — Module 2 : Tarification                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Q : "C'est quoi vᵗ et à quoi ça sert ?"                            ║
║  R : "vᵗ = 1/(1+i)ᵗ est le facteur d'actualisation. Il traduit      ║
║       le fait qu'1€ aujourd'hui vaut plus qu'1€ dans t ans.          ║
║       Dans les provisions, chaque flux futur est multiplié           ║
║       par vᵗ pour ramener sa valeur à aujourd'hui."                  ║
║                                                                      ║
║  Q : "Comment calculez-vous la prime d'une rente ?"                  ║
║  R : "La prime pure unique est äx = Σ S(t) × vᵗ.                    ║
║       Pour une rente de 14 400€/an à 65 ans avec i=2%,              ║
║       j'obtiens ~218 000€. On ajoute ensuite les chargements         ║
║       acquisition (α=5%) → prime nette ~230 000€."                   ║
║                                                                      ║
║  Q : "Pourquoi le taux technique est-il réglementé ?"                ║
║  R : "L'assureur garantit un rendement i à l'assuré. Si              ║
║       les marchés délivrent moins, il est en déficit.                ║
║       L'ACPR plafonne i pour limiter ce risque de taux —             ║
║       c'est le cœur de la gestion actif-passif."                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("🎉 MODULE 2 COMPLET — 4 notions maîtrisées :")
print("   ✅ Notion 5 : vᵗ — taux technique et actualisation")
print("   ✅ Notion 6 : Prime pure assurance décès temporaire")
print("   ✅ Notion 7 : Prime pure rente viagère (äx)")
print("   ✅ Notion 8 : Chargements et prime nette")
print()
print("→ Prochain module : Module 3 — Provisions Mathématiques (PM)")
print("  (Best Estimate, rachat, courbe des taux — Leçons 6-7)")
