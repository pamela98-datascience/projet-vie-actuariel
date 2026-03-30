"""
Module 5 - Backtesting PM (Analyse N/N+1)
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)

Contenu (= Module 5 des notebooks Colab) :
- Construction du portefeuille de contrats
- Simulation des événements N/N+1
- PM attendue vs PM réelle
- Décomposition écart : effet taux / mortalité / portefeuille
- Rapport de backtesting
"""

import numpy as np
import pandas as pd


def construire_portefeuille(n_contrats: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Génère un portefeuille de contrats avec âges et rentes hétérogènes.
    """
    np.random.seed(seed)
    ages = np.random.choice(
        [62, 63, 64, 65, 66, 67, 68],
        size=n_contrats,
        p=[0.05, 0.10, 0.20, 0.35, 0.15, 0.10, 0.05]
    )
    rentes = np.random.choice(
        [9_600, 12_000, 14_400, 18_000, 24_000],
        size=n_contrats,
        p=[0.20, 0.30, 0.30, 0.15, 0.05]
    )
    return pd.DataFrame({
        "id_contrat": range(n_contrats),
        "age_N":      ages,
        "rente":      rentes,
    })


def calculer_PM_portefeuille(portefeuille: pd.DataFrame, df_table: pd.DataFrame,
                              PM_fn, taux_i: float) -> pd.DataFrame:
    """Calcule la PM pour chaque contrat du portefeuille."""
    port = portefeuille.copy()
    port["PM_N"] = port.apply(
        lambda r: PM_fn(df_table, int(r["age_N"]), r["rente"], taux_i), axis=1
    )
    return port


def simuler_evenements(portefeuille: pd.DataFrame, df_table: pd.DataFrame,
                       facteur_mortalite: float = 0.80,
                       taux_rachat: float = 0.04,
                       seed: int = 123) -> pd.DataFrame:
    """
    Simule les événements entre N et N+1 pour chaque contrat.
    Retourne le portefeuille avec colonnes deces / rachat / survie.
    """
    np.random.seed(seed)
    port = portefeuille.copy()
    evenements = []

    for _, row in port.iterrows():
        age     = int(row["age_N"])
        qx_reel = df_table.loc[df_table["age"] == age, "qx"].values[0] * facteur_mortalite
        deces   = np.random.random() < qx_reel
        rachat  = (not deces) and (np.random.random() < taux_rachat)
        survie  = not deces and not rachat
        evenements.append({"id_contrat": row["id_contrat"],
                           "deces": deces, "rachat": rachat, "survie": survie})

    df_evt = pd.DataFrame(evenements)
    return port.merge(df_evt, on="id_contrat")


def backtesting_N_N1(portefeuille: pd.DataFrame, df_table: pd.DataFrame,
                     PM_fn, taux_N: float, taux_N1: float) -> pd.DataFrame:
    """
    Calcule PM attendue et PM réelle à N+1 pour les contrats survivants.

    PM attendue : recalcul à age+1 avec taux_N (hypothèses inchangées)
    PM réelle   : recalcul à age+1 avec taux_N1 (nouveau taux de marché)
    """
    port_survie = portefeuille[portefeuille["survie"]].copy()

    port_survie["PM_N1_attendue"] = port_survie.apply(
        lambda r: PM_fn(df_table, int(r["age_N"]) + 1, r["rente"], taux_N), axis=1
    )
    port_survie["PM_N1_reelle"] = port_survie.apply(
        lambda r: PM_fn(df_table, int(r["age_N"]) + 1, r["rente"], taux_N1), axis=1
    )
    port_survie["ecart_absolu"]  = (port_survie["PM_N1_reelle"]
                                    - port_survie["PM_N1_attendue"])
    port_survie["ecart_relatif"] = (port_survie["ecart_absolu"]
                                    / port_survie["PM_N1_attendue"] * 100)
    return port_survie


def decomposer_ecart(portefeuille: pd.DataFrame, df_table: pd.DataFrame,
                     PM_fn, taux_N: float, taux_N1: float,
                     facteur_mortalite: float, n_deces_reels: int,
                     n_rachats_reels: int, n_rachats_attendus: float) -> dict:
    """
    Décompose l'écart total en 3 effets :
    - Effet taux      : changement de la courbe de taux
    - Effet mortalité : décès réels vs attendus
    - Effet portefeuille : rachats réels vs attendus (résiduel)
    """
    port_survie   = portefeuille[portefeuille["survie"]].copy()
    total_att     = port_survie.apply(
        lambda r: PM_fn(df_table, int(r["age_N"]) + 1, r["rente"], taux_N), axis=1
    ).sum()
    total_reel    = port_survie.apply(
        lambda r: PM_fn(df_table, int(r["age_N"]) + 1, r["rente"], taux_N1), axis=1
    ).sum()
    ecart_total   = total_reel - total_att

    # Effet taux : PM avec nouveau taux sur même portefeuille survivant
    effet_taux = total_reel - total_att

    # Effet mortalité : décès attendus vs réels × PM moyenne libérée
    deces_att       = portefeuille.apply(
        lambda r: df_table.loc[df_table["age"]==int(r["age_N"]), "qx"].values[0]
                  * facteur_mortalite, axis=1
    ).sum()
    PM_moy          = portefeuille["PM_N"].mean()
    effet_mortalite = (deces_att - n_deces_reels) * PM_moy

    # Effet portefeuille : résiduel
    effet_portefeuille = ecart_total - effet_taux - effet_mortalite

    return {
        "PM attendue N+1 (€)":        round(total_att, 0),
        "PM réelle N+1 (€)":          round(total_reel, 0),
        "Écart total (€)":            round(ecart_total, 0),
        "Écart total (%)":            round(ecart_total / total_att * 100, 2),
        "Effet taux (€)":             round(effet_taux, 0),
        "Effet taux (%)":             round(effet_taux / abs(ecart_total) * 100 if ecart_total != 0 else 0, 1),
        "Effet mortalité (€)":        round(effet_mortalite, 0),
        "Effet mortalité (%)":        round(effet_mortalite / abs(ecart_total) * 100 if ecart_total != 0 else 0, 1),
        "Effet portefeuille (€)":     round(effet_portefeuille, 0),
        "Effet portefeuille (%)":     round(effet_portefeuille / abs(ecart_total) * 100 if ecart_total != 0 else 0, 1),
    }


def analyse_par_age(port_survie: pd.DataFrame) -> pd.DataFrame:
    """Analyse des écarts par tranche d'âge."""
    port_survie = port_survie.copy()
    port_survie["tranche"] = pd.cut(
        port_survie["age_N"],
        bins=[61, 64, 66, 68, 110],
        labels=["62-64 ans", "65-66 ans", "67-68 ans", "69+ ans"]
    )
    return port_survie.groupby("tranche", observed=True).agg(
        n_contrats    = ("id_contrat", "count"),
        PM_att        = ("PM_N1_attendue", "sum"),
        PM_reel       = ("PM_N1_reelle", "sum"),
        ecart_moy_pct = ("ecart_relatif", "mean"),
    ).reset_index().assign(
        ecart_total = lambda d: d["PM_reel"] - d["PM_att"],
        ecart_pct   = lambda d: (d["PM_reel"] - d["PM_att"]) / d["PM_att"] * 100
    )


def rapport_backtesting(decomp: dict, n_contrats: int, n_deces: int,
                        n_rachats: int, n_survie: int,
                        taux_N: float, taux_N1: float,
                        facteur_mortalite: float,
                        taux_rachat_reel: float,
                        taux_rachat_att: float) -> str:
    """Génère le rapport textuel de backtesting."""
    sous_prov = "SOUS-PROVISIONNEMENT" if decomp["Écart total (€)"] > 0 else "SUR-PROVISIONNEMENT"
    action1 = f"Mettre à jour la courbe des taux ({taux_N*100:.1f}% → {taux_N1*100:.1f}%)"
    action2 = (f"Surveiller la mortalité sur 2-3 ans "
               f"(réelle à {facteur_mortalite*100:.0f}% de la table)")
    action3 = (f"Réviser le taux de rachat de {taux_rachat_att*100:.0f}% "
               f"à {taux_rachat_reel*100:.0f}%")

    return f"""
PÉRIMÈTRE
  Contrats analysés : {n_contrats}  |  Décès : {n_deces} ({n_deces/n_contrats*100:.1f}%)
  Rachats : {n_rachats} ({n_rachats/n_contrats*100:.1f}%)  |  Survivants : {n_survie}

ÉCART GLOBAL : {sous_prov}
  PM attendue à N+1   : {decomp['PM attendue N+1 (€)']:>14,.0f} €
  PM réelle à N+1     : {decomp['PM réelle N+1 (€)']:>14,.0f} €
  Écart total         : {decomp['Écart total (€)']:>+14,.0f} € ({decomp['Écart total (%)']:+.2f}%)

DÉCOMPOSITION
  Effet taux          : {decomp['Effet taux (€)']:>+14,.0f} € ({decomp['Effet taux (%)']:+.1f}%)
  Effet mortalité     : {decomp['Effet mortalité (€)']:>+14,.0f} € ({decomp['Effet mortalité (%)']:+.1f}%)
  Effet portefeuille  : {decomp['Effet portefeuille (€)']:>+14,.0f} € ({decomp['Effet portefeuille (%)']:+.1f}%)

RECOMMANDATIONS
  PRIORITÉ 1 : {action1}
  PRIORITÉ 2 : {action2}
  PRIORITÉ 3 : {action3}
"""
