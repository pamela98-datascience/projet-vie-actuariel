"""
Module 2 - Tarification
Projet Vie Actuariel - Pamela Fagla (ISUP M1 Actuariat)

Contenu (= Notions 5, 6, 7, 8 des notebooks Colab) :
- Facteur d'actualisation vt
- Prime pure assurance décès temporaire
- Prime pure rente viagère (annuité actuarielle äx)
- Chargements et prime nette
"""

import numpy as np
import pandas as pd


def facteur_actualisation(taux_i: float, t: int) -> float:
    """vt = 1 / (1+i)^t"""
    if t == 0:
        return 1.0
    return 1 / (1 + taux_i) ** t


def tableau_vt(taux_dict: dict, horizons: list) -> pd.DataFrame:
    """
    Tableau comparatif des facteurs vt selon différents taux.
    taux_dict = {"i=2%": 0.02, "i=3%": 0.03, ...}
    """
    rows = []
    for t in horizons:
        row = {"t (années)": t}
        for nom, i in taux_dict.items():
            row[nom] = round(facteur_actualisation(i, t), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def prime_pure_deces(df: pd.DataFrame, age_x: int, n_ans: int,
                     capital_C: float, taux_i: float) -> tuple:
    """
    Prime pure unique d'une assurance décès temporaire n ans.

    PP = C × Σ(t=0 à n-1) [ tpx × qx+t × v^(t+1) ]

    Retourne : (prime_pure, DataFrame des flux annuels)
    """
    v  = 1 / (1 + taux_i)
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
        tpx  = lxt / lx
        qxt  = df.loc[df["age"] == age_t, "qx"].values[0]
        prob_deces_t = tpx * qxt
        vt1  = v ** (t + 1)
        flux = capital_C * prob_deces_t * vt1
        pp  += flux

        flux_list.append({
            "t": t,
            "Âge": age_t,
            "tpx": round(tpx, 6),
            "qx+t (‰)": round(qxt * 1000, 4),
            "Prob. décès": round(prob_deces_t, 6),
            "v^(t+1)": round(vt1, 6),
            "Flux (€)": round(flux, 2),
        })

    return pp, pd.DataFrame(flux_list)


def annuite_actuarielle(df: pd.DataFrame, age_x: int,
                        taux_i: float, rente_R: float = 1.0) -> tuple:
    """
    Annuité actuarielle äx = Σ S(t) × vt
    = valeur d'une rente viagère de rente_R €/an depuis age_x.

    Retourne : (äx × rente_R, DataFrame des flux)
    """
    v   = 1 / (1 + taux_i)
    lx  = df.loc[df["age"] == age_x, "lx"].values[0]
    a_x = 0.0
    flux_list = []

    for t in range(111 - age_x):
        age_t = age_x + t
        if age_t > 110:
            break
        St   = df.loc[df["age"] == age_t, "lx"].values[0] / lx
        vt   = v ** t
        flux = rente_R * St * vt
        a_x += flux

        flux_list.append({
            "t": t,
            "Âge": age_t,
            "S(t)": round(St, 5),
            "vᵗ": round(vt, 5),
            "S(t)×vᵗ": round(St * vt, 5),
            "Flux (€/an)": round(flux, 2),
        })

    return a_x, pd.DataFrame(flux_list)


def prime_nette(prime_pure: float, alpha: float) -> float:
    """
    Prime nette unique après chargement acquisition.
    Prime nette = PP / (1 - alpha)
    """
    return prime_pure / (1 - alpha)


def sensibilite_taux(df: pd.DataFrame, age_x: int, rente_R: float,
                     taux_list: list) -> pd.DataFrame:
    """
    Tableau de sensibilité de la prime pure rente selon le taux technique.
    """
    rows = []
    for i in taux_list:
        pp, _ = annuite_actuarielle(df, age_x, i, rente_R)
        rows.append({
            "Taux i": f"{i*100:.1f}%",
            "äx": round(pp / rente_R, 4),
            "Prime pure (€)": round(pp, 2),
        })
    return pd.DataFrame(rows)


def comparaison_HF(df_h: pd.DataFrame, df_f: pd.DataFrame,
                   age_x: int, rente_R: float, taux_i: float) -> pd.DataFrame:
    """Compare prime pure Homme vs Femme pour la même rente."""
    pp_h, _ = annuite_actuarielle(df_h, age_x, taux_i, rente_R)
    pp_f, _ = annuite_actuarielle(df_f, age_x, taux_i, rente_R)
    return pd.DataFrame([
        {"Table": "TH 00-02 (Homme)", "äx": round(pp_h/rente_R, 4),
         "Prime pure (€)": round(pp_h, 2), "Surcoût vs Homme (€)": 0},
        {"Table": "TF 00-02 (Femme)", "äx": round(pp_f/rente_R, 4),
         "Prime pure (€)": round(pp_f, 2),
         "Surcoût vs Homme (€)": round(pp_f - pp_h, 2)},
    ])
