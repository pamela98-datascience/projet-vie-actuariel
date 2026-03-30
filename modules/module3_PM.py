"""
Module 3 - Provisions Mathématiques (PM)
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)

Contenu (= Notions 9, 10, 11, 12 des notebooks Colab) :
- PM prospective sur rente viagère
- Évolution de la PM dans le temps (récurrence de Thiele)
- Valeur de rachat (total, partiel)
- Best Estimate S2 (courbe des taux Nelson-Siegel, SCR taux)
"""

import numpy as np
import pandas as pd


def PM_prospective(df: pd.DataFrame, age_x: int,
                   rente_R: float, taux_i: float) -> float:
    """
    PM prospective = R × äx+t
    = Σ S(t) × vᵗ × R depuis l'âge age_x.
    """
    v  = 1 / (1 + taux_i)
    lx = df.loc[df["age"] == age_x, "lx"].values[0]
    pm = 0.0
    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110:
            break
        St = df.loc[df["age"] == age_t, "lx"].values[0] / lx
        pm += rente_R * St * (v ** t)
    return pm


def PM_flux_detail(df: pd.DataFrame, age_x: int,
                   rente_R: float, taux_i: float) -> pd.DataFrame:
    """
    Détail des flux annuels constituant la PM.
    Retourne un DataFrame avec S(t), vt, flux, PM cumulée.
    """
    v   = 1 / (1 + taux_i)
    lx  = df.loc[df["age"] == age_x, "lx"].values[0]
    rows = []
    PM_cumul = 0.0

    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110:
            break
        St   = df.loc[df["age"] == age_t, "lx"].values[0] / lx
        vt   = v ** t
        flux = rente_R * St * vt
        PM_cumul += flux
        rows.append({
            "t": t, "Âge": age_t,
            "S(t)": round(St, 5),
            "vᵗ": round(vt, 5),
            "Flux R×S(t)×vᵗ (€)": round(flux, 2),
            "PM cumulée (€)": round(PM_cumul, 2),
        })

    return pd.DataFrame(rows)


def evolution_PM(df: pd.DataFrame, age_depart: int,
                 rente_R: float, taux_i: float,
                 n_annees: int = 40) -> pd.DataFrame:
    """
    Évolution de la PM année par année (méthode prospective).
    Vérifie aussi la récurrence de Thiele.
    """
    PM_0   = PM_prospective(df, age_depart, rente_R, taux_i)
    lx_dep = df.loc[df["age"] == age_depart, "lx"].values[0]
    rows   = []

    for t in range(n_annees + 1):
        age_t = age_depart + t
        if age_t > 105:
            break
        PM_t = PM_prospective(df, age_t, rente_R, taux_i)
        St   = df.loc[df["age"] == age_t, "lx"].values[0] / lx_dep
        rows.append({
            "t": t,
            "Âge": age_t,
            "S(t)": round(St, 5),
            "PM (€)": round(PM_t, 2),
            "PM / PM₀ (%)": round(PM_t / PM_0 * 100, 2),
        })

    return pd.DataFrame(rows)


def thiele_verification(df: pd.DataFrame, age_depart: int,
                        rente_R: float, taux_i: float,
                        n_annees: int = 5) -> pd.DataFrame:
    """
    Vérifie la récurrence de Thiele :
    PM(t+1) = [PM(t) × (1+i) − R] × px+t
    Compare avec le calcul prospectif.
    """
    rows  = []
    PM_th = PM_prospective(df, age_depart, rente_R, taux_i)

    for t in range(n_annees):
        age_t   = age_depart + t
        px_t    = df.loc[df["age"] == age_t, "px"].values[0]
        PM_new  = (PM_th * (1 + taux_i) - rente_R) * px_t
        PM_ref  = PM_prospective(df, age_depart + t + 1, rente_R, taux_i)
        rows.append({
            "t": t,
            "PM(t) Thiele (€)": round(PM_th, 2),
            "PM(t+1) Thiele (€)": round(PM_new, 2),
            "PM(t+1) Prospectif (€)": round(PM_ref, 2),
            "Écart (€)": round(abs(PM_new - PM_ref), 2),
        })
        PM_th = PM_new

    return pd.DataFrame(rows)


def valeur_rachat(PM: float, taux_penalite: float,
                  pct_rachat: float = 1.0) -> dict:
    """
    Valeur de rachat total ou partiel.
    pct_rachat = 1.0 pour rachat total, 0.2 pour rachat partiel 20%
    """
    montant_rachete = PM * pct_rachat * (1 - taux_penalite)
    penalite        = PM * pct_rachat * taux_penalite
    PM_residuelle   = PM * (1 - pct_rachat)
    return {
        "PM initiale (€)":      round(PM, 2),
        "% racheté":            round(pct_rachat * 100, 1),
        "Montant versé (€)":    round(montant_rachete, 2),
        "Pénalité (€)":         round(penalite, 2),
        "PM résiduelle (€)":    round(PM_residuelle, 2),
    }


def courbe_nelson_siegel(t: float, beta0: float = 0.03,
                         beta1: float = -0.02, tau: float = 5.0) -> float:
    """
    Courbe des taux zéro-coupon Nelson-Siegel.
    beta0 = niveau long terme
    beta1 = pente (spread court/long)
    tau   = paramètre de forme
    """
    if t <= 0:
        return beta0 + beta1
    facteur = (1 - np.exp(-t / tau)) / (t / tau)
    return beta0 + beta1 * facteur


def best_estimate(df: pd.DataFrame, age_x: int,
                  rente_R: float, courbe_fn) -> float:
    """
    Best Estimate S2 avec courbe des taux quelconque.
    BE = Σ flux_t × S(t) × vᵗ(courbe)
    """
    lx = df.loc[df["age"] == age_x, "lx"].values[0]
    BE = 0.0
    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110:
            break
        St  = df.loc[df["age"] == age_t, "lx"].values[0] / lx
        i_t = courbe_fn(t)
        vt  = 1 / (1 + i_t) ** t if t > 0 else 1.0
        BE += rente_R * St * vt
    return BE


def scr_taux(df: pd.DataFrame, age_x: int,
             rente_R: float, taux_base: float,
             choc_bps: float = 100) -> dict:
    """
    SCR taux S2 : choc ±choc_bps points de base sur courbe plate.
    Retourne BE base, BE down, BE up et SCR.
    """
    choc = choc_bps / 10000
    BE_base = best_estimate(df, age_x, rente_R, lambda t: taux_base)
    BE_down = best_estimate(df, age_x, rente_R, lambda t: max(0, taux_base - choc))
    BE_up   = best_estimate(df, age_x, rente_R, lambda t: taux_base + choc)
    SCR     = max(BE_down - BE_base, BE_base - BE_up)
    return {
        "BE base (€)":    round(BE_base, 2),
        "BE choc down (€)": round(BE_down, 2),
        "BE choc up (€)":   round(BE_up, 2),
        f"ΔBE down (+{choc_bps}bps)": round(BE_down - BE_base, 2),
        f"ΔBE up (-{choc_bps}bps)":   round(BE_base - BE_up, 2),
        "SCR taux (€)":   round(SCR, 2),
    }
