"""
Module 4 - Inventaire & Résultats
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)

Contenu (= Module 4 des notebooks Colab) :
- Inventaire trimestriel : équation de passage PM
- Décomposition résultat : financier / mortalité / gestion
- Participation aux bénéfices (PB, PPB)
- Duration et gestion actif-passif (ALM)
"""

import numpy as np
import pandas as pd


def inventaire_annuel(PM_deb: float, n_assures: int, age_moy: int,
                      rendement_reel: float, tx_mortalite_reel: float,
                      frais_reels_pct: float, tx_rachat_reel: float,
                      rente_R: float, taux_i: float,
                      taux_gestion: float, taux_penalite: float,
                      df_table, PM_fn) -> dict:
    """
    Calcule l'inventaire d'une année et décompose le résultat.

    Paramètres :
      PM_deb           : PM portefeuille en début d'année
      n_assures        : nombre d'assurés en début d'année
      age_moy          : âge moyen du portefeuille
      rendement_reel   : rendement réel des placements
      tx_mortalite_reel: ratio décès réels / attendus (ex : 0.85)
      frais_reels_pct  : frais réels en % de la PM
      tx_rachat_reel   : taux de rachat réel
      rente_R          : rente annuelle par assuré
      taux_i           : taux technique garanti
      taux_gestion     : chargements gestion (β)
      taux_penalite    : pénalité sur rachats
      df_table         : table de mortalité
      PM_fn            : fonction PM_prospective(df, age, rente, taux)
    """
    PM_par_tete = PM_deb / n_assures if n_assures > 0 else 0

    # ── Décès ──────────────────────────────────────────────
    qx_att      = df_table.loc[df_table["age"] == age_moy, "qx"].values[0]
    deces_att   = n_assures * qx_att
    deces_reels = deces_att * tx_mortalite_reel

    # ── Rachats ─────────────────────────────────────────────
    n_rachats    = n_assures * tx_rachat_reel
    rachats_tot  = n_rachats * PM_par_tete * (1 - taux_penalite)
    penalites    = n_rachats * PM_par_tete * taux_penalite

    # ── Assurés fin d'année ─────────────────────────────────
    n_fin   = max(n_assures - deces_reels - n_rachats, 0)
    n_moyen = (n_assures + n_fin) / 2

    # ── Rentes versées ──────────────────────────────────────
    rentes_tot = n_moyen * rente_R

    # ── Produits financiers ─────────────────────────────────
    prod_fin_reels = PM_deb * rendement_reel
    prod_fin_att   = PM_deb * taux_i
    res_financier  = prod_fin_reels - prod_fin_att

    # ── Résultat mortalité ──────────────────────────────────
    # Moins de décès réels = risque longévité = perte pour rentes
    surplus_deces = deces_att - deces_reels
    res_mortalite = -surplus_deces * PM_par_tete

    # ── Résultat gestion ────────────────────────────────────
    chargements   = PM_deb * taux_gestion
    frais_reels   = PM_deb * frais_reels_pct
    res_gestion   = chargements - frais_reels

    # ── Résultat total ──────────────────────────────────────
    res_total = res_financier + res_mortalite + res_gestion

    # ── PB minimale légale ──────────────────────────────────
    PB_fin  = max(0, 0.85 * res_financier)
    PB_tech = max(0, 0.90 * (res_mortalite + res_gestion))
    PB_tot  = PB_fin + PB_tech

    # ── PM fin d'année ──────────────────────────────────────
    age_fin        = min(age_moy + 1, 105)
    PM_par_tete_fin = PM_fn(df_table, age_fin, rente_R, taux_i)
    PM_fin          = PM_par_tete_fin * n_fin

    return {
        "n_debut":          int(n_assures),
        "n_fin":            int(n_fin),
        "deces_att":        round(deces_att, 1),
        "deces_reels":      round(deces_reels, 1),
        "n_rachats":        round(n_rachats, 1),
        "PM_deb":           round(PM_deb, 0),
        "PM_fin":           round(PM_fin, 0),
        "rentes_versees":   round(rentes_tot, 0),
        "prod_fin_reels":   round(prod_fin_reels, 0),
        "prod_fin_att":     round(prod_fin_att, 0),
        "res_financier":    round(res_financier, 0),
        "res_mortalite":    round(res_mortalite, 0),
        "res_gestion":      round(res_gestion, 0),
        "res_total":        round(res_total, 0),
        "PB_financiere":    round(PB_fin, 0),
        "PB_technique":     round(PB_tech, 0),
        "PB_totale":        round(PB_tot, 0),
        "penalites":        round(penalites, 0),
    }


def simulation_multi_annees(df_table, PM_fn, age_0: int, rente_R: float,
                             taux_i: float, n_assures_0: int,
                             taux_gestion: float, taux_penalite: float,
                             scenarios: list) -> pd.DataFrame:
    """
    Simule l'inventaire sur plusieurs années.
    scenarios = liste de dicts avec les réalisations annuelles.
    """
    PM_0       = PM_fn(df_table, age_0, rente_R, taux_i)
    PM_courant = PM_0 * n_assures_0
    n_courant  = n_assures_0
    age_courant = age_0
    rows = []

    for sc in scenarios:
        res = inventaire_annuel(
            PM_courant, n_courant, age_courant,
            sc["rendement_reel"], sc["tx_mortalite_reel"],
            sc["frais_reels_pct"], sc["tx_rachat_reel"],
            rente_R, taux_i, taux_gestion, taux_penalite,
            df_table, PM_fn
        )
        row = {
            "Année":             sc["annee"],
            "Rendement réel":    f"{sc['rendement_reel']*100:.1f}%",
            "Mortalité (% att.)": f"{sc['tx_mortalite_reel']*100:.0f}%",
            "Assurés début":     res["n_debut"],
            "Décès attendus":    res["deces_att"],
            "Décès réels":       res["deces_reels"],
            "Rachats":           res["n_rachats"],
            "Assurés fin":       res["n_fin"],
            "PM début (€)":      res["PM_deb"],
            "PM fin (€)":        res["PM_fin"],
            "Rentes versées (€)": res["rentes_versees"],
            "Rés. financier (€)": res["res_financier"],
            "Rés. mortalité (€)": res["res_mortalite"],
            "Rés. gestion (€)":  res["res_gestion"],
            "Rés. total (€)":    res["res_total"],
            "PB totale (€)":     res["PB_totale"],
        }
        rows.append(row)
        PM_courant  = res["PM_fin"]
        n_courant   = res["n_fin"]
        age_courant += 1

    return pd.DataFrame(rows)


def duration_PM(df: pd.DataFrame, age_x: int,
                rente_R: float, taux_i: float) -> float:
    """
    Duration de Macaulay de la PM.
    D = Σ(t × flux_t) / Σ(flux_t)
    """
    v   = 1 / (1 + taux_i)
    lx  = df.loc[df["age"] == age_x, "lx"].values[0]
    num = 0.0
    den = 0.0
    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110:
            break
        St   = df.loc[df["age"] == age_t, "lx"].values[0] / lx
        flux = rente_R * St * (v ** t)
        num += t * flux
        den += flux
    return num / den if den > 0 else 0


def tableau_duration(df: pd.DataFrame, ages: list,
                     rente_R: float, taux_i: float) -> pd.DataFrame:
    """Tableau des durations pour différents âges."""
    rows = []
    for a in ages:
        if a > 105:
            break
        D    = duration_PM(df, a, rente_R, taux_i)
        D_mod = D / (1 + taux_i)
        rows.append({
            "Âge": a,
            "Duration (ans)":           round(D, 2),
            "Duration modifiée":        round(D_mod, 2),
            "Sensibilité −1% taux (%)": round(D_mod, 2),
        })
    return pd.DataFrame(rows)


def pb_detail(res_financier: float, res_mortalite: float,
              res_gestion: float, n_assures: int) -> dict:
    """Détail de la participation aux bénéfices."""
    PB_fin  = max(0, 0.85 * res_financier)
    PB_tech = max(0, 0.90 * (res_mortalite + res_gestion))
    PB_tot  = PB_fin + PB_tech
    PB_unit = PB_tot / n_assures if n_assures > 0 else 0
    return {
        "Résultat financier (€)":      round(res_financier, 0),
        "PB financière (85%) (€)":     round(PB_fin, 0),
        "Résultat technique (€)":      round(res_mortalite + res_gestion, 0),
        "PB technique (90%) (€)":      round(PB_tech, 0),
        "PB TOTALE (€)":               round(PB_tot, 0),
        "PB par assuré (€)":           round(PB_unit, 2),
    }
