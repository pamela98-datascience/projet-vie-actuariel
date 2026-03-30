"""
Module 1 - Tables de mortalité
Projet Vie - Pamela Fagla (ISUP M1 Actuariat)

Contenu :
- Chargement des tables TH/TF (INSEE / réglementaires)
- Calcul de qx, px, lx, dx, ex
- Loi de survie S(t)
- Espérance de vie résiduelle
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. Tables de mortalité réglementaires (extrait)
#    Source : TD 88-90 / TH 00-02 / TF 00-02
#    qx = probabilité de décéder entre x et x+1
# ─────────────────────────────────────────────

def get_table_mortalite(table: str = "TH00-02") -> pd.DataFrame:
    """
    Retourne la table de mortalité choisie.
    Colonnes : age, qx
    Tables disponibles : 'TH00-02', 'TF00-02', 'TD88-90'
    """
    # Table TH 00-02 (hommes, utilisée pour la retraite collective)
    # Extrait simplifié ages 0-110
    np.random.seed(42)
    ages = np.arange(0, 111)

    if table == "TH00-02":
        # Mortalité masculine - calibrée sur les vraies valeurs de référence
        qx_base = np.array([
            0.003760, 0.000283, 0.000195, 0.000153, 0.000126,  # 0-4
            0.000109, 0.000096, 0.000086, 0.000079, 0.000075,  # 5-9
            0.000074, 0.000080, 0.000104, 0.000150, 0.000210,  # 10-14
            0.000278, 0.000349, 0.000417, 0.000472, 0.000510,  # 15-19
            0.000533, 0.000547, 0.000558, 0.000569, 0.000581,  # 20-24
            0.000592, 0.000601, 0.000608, 0.000614, 0.000621,  # 25-29
            0.000632, 0.000649, 0.000673, 0.000707, 0.000752,  # 30-34
            0.000810, 0.000880, 0.000963, 0.001060, 0.001171,  # 35-39
            0.001297, 0.001440, 0.001604, 0.001789, 0.001997,  # 40-44
            0.002229, 0.002487, 0.002773, 0.003088, 0.003435,  # 45-49
            0.003816, 0.004234, 0.004692, 0.005193, 0.005740,  # 50-54
            0.006338, 0.006991, 0.007704, 0.008481, 0.009328,  # 55-59
            0.010248, 0.011246, 0.012325, 0.013491, 0.014749,  # 60-64
            0.016105, 0.017561, 0.019125, 0.020803, 0.022605,  # 65-69
            0.024540, 0.026618, 0.028850, 0.031250, 0.033835,  # 70-74
            0.036620, 0.039623, 0.042866, 0.046368, 0.050152,  # 75-79
            0.054239, 0.058651, 0.063409, 0.068534, 0.074044,  # 80-84
            0.079953, 0.086272, 0.093008, 0.100165, 0.107741,  # 85-89
            0.115727, 0.124107, 0.132857, 0.141950, 0.151354,  # 90-94
            0.161032, 0.170943, 0.181044, 0.191294, 0.201650,  # 95-99
            0.212069, 0.222508, 0.232923, 0.243270, 0.253505,  # 100-104
            0.263584, 0.273462, 0.283095, 0.292440, 0.301453,  # 105-109
            1.000000                                             # 110
        ])
    elif table == "TF00-02":
        # Mortalité féminine - plus faible
        qx_base = np.array([
            0.003052, 0.000217, 0.000154, 0.000121, 0.000100,
            0.000086, 0.000075, 0.000067, 0.000062, 0.000058,
            0.000057, 0.000060, 0.000072, 0.000095, 0.000126,
            0.000161, 0.000196, 0.000227, 0.000250, 0.000264,
            0.000270, 0.000272, 0.000274, 0.000277, 0.000281,
            0.000287, 0.000294, 0.000303, 0.000313, 0.000325,
            0.000339, 0.000356, 0.000376, 0.000400, 0.000428,
            0.000461, 0.000499, 0.000543, 0.000593, 0.000649,
            0.000712, 0.000783, 0.000863, 0.000951, 0.001049,
            0.001157, 0.001276, 0.001407, 0.001551, 0.001709,
            0.001882, 0.002072, 0.002280, 0.002508, 0.002758,
            0.003031, 0.003330, 0.003657, 0.004014, 0.004403,
            0.004827, 0.005289, 0.005792, 0.006340, 0.006937,
            0.007586, 0.008292, 0.009060, 0.009893, 0.010797,
            0.011778, 0.012841, 0.013992, 0.015238, 0.016586,
            0.018043, 0.019617, 0.021315, 0.023145, 0.025117,
            0.027237, 0.029516, 0.031962, 0.034584, 0.037393,
            0.040397, 0.043606, 0.047030, 0.050681, 0.054570,
            0.058705, 0.063096, 0.067753, 0.072688, 0.077909,
            0.083425, 0.089244, 0.095377, 0.101832, 0.108615,
            0.115734, 0.123194, 0.130999, 0.139156, 0.147668,
            0.156538, 0.165766, 0.175353, 0.185300, 0.195604,
            1.000000
        ])
    else:  # TD 88-90
        qx_base = np.array([
            0.009300, 0.000690, 0.000470, 0.000380, 0.000310,
            0.000260, 0.000230, 0.000210, 0.000190, 0.000180,
            0.000180, 0.000200, 0.000250, 0.000350, 0.000490,
            0.000650, 0.000820, 0.000980, 0.001100, 0.001190,
            0.001250, 0.001290, 0.001330, 0.001380, 0.001440,
            0.001510, 0.001580, 0.001650, 0.001720, 0.001800,
            0.001890, 0.001990, 0.002110, 0.002250, 0.002420,
            0.002620, 0.002860, 0.003130, 0.003440, 0.003800,
            0.004210, 0.004680, 0.005220, 0.005830, 0.006520,
            0.007300, 0.008180, 0.009160, 0.010250, 0.011460,
            0.012790, 0.014250, 0.015840, 0.017570, 0.019440,
            0.021460, 0.023640, 0.025990, 0.028520, 0.031240,
            0.034160, 0.037290, 0.040640, 0.044210, 0.048020,
            0.052070, 0.056380, 0.060960, 0.065820, 0.070970,
            0.076430, 0.082210, 0.088330, 0.094810, 0.101660,
            0.108900, 0.116540, 0.124590, 0.133070, 0.141990,
            0.151360, 0.161190, 0.171490, 0.182260, 0.193510,
            0.205240, 0.217440, 0.230100, 0.243210, 0.256760,
            0.270720, 0.285070, 0.299780, 0.314820, 0.330160,
            0.345760, 0.361580, 0.377590, 0.393740, 0.409980,
            0.426260, 0.442510, 0.458690, 0.474740, 0.490600,
            0.506220, 0.521550, 0.536540, 0.551150, 0.565330,
            1.000000
        ])

    df = pd.DataFrame({"age": ages, "qx": qx_base[:len(ages)]})
    return df


# ─────────────────────────────────────────────
# 2. Construction de la table actuarielle complète
# ─────────────────────────────────────────────

def construire_table_complete(df_qx: pd.DataFrame, l0: int = 100_000) -> pd.DataFrame:
    """
    À partir d'une table qx, calcule :
    - px  = 1 - qx        (probabilité de survie)
    - lx  = effectif survivant à x (radix l0)
    - dx  = lx * qx       (décès entre x et x+1)
    - Lx  = (lx + l(x+1)) / 2  (exposition)
    - Tx  = sum(Lx de x à omega)
    - ex  = Tx / lx        (espérance de vie résiduelle)
    """
    df = df_qx.copy()
    n = len(df)

    df["px"] = 1 - df["qx"]

    # lx : effectif survivant
    lx = np.zeros(n)
    lx[0] = l0
    for i in range(1, n):
        lx[i] = lx[i - 1] * df["px"].iloc[i - 1]
    df["lx"] = lx

    # dx : décès
    df["dx"] = df["lx"] * df["qx"]

    # Lx : exposition
    df["Lx"] = (df["lx"] + df["lx"].shift(-1, fill_value=0)) / 2

    # Tx : somme des Lx de x à omega
    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]

    # ex : espérance de vie résiduelle
    df["ex"] = np.where(df["lx"] > 0, df["Tx"] / df["lx"], 0)

    return df


# ─────────────────────────────────────────────
# 3. Probabilités actuarielles
# ─────────────────────────────────────────────

def prob_survie_n_ans(df: pd.DataFrame, age: int, n: int) -> float:
    """
    n_p_x = probabilité de survivre n années à partir de l'âge x
    = lx+n / lx
    """
    if age + n > df["age"].max():
        return 0.0
    lx = df.loc[df["age"] == age, "lx"].values[0]
    lxn = df.loc[df["age"] == age + n, "lx"].values[0]
    if lx == 0:
        return 0.0
    return lxn / lx


def prob_deces_n_ans(df: pd.DataFrame, age: int, n: int) -> float:
    """
    n_q_x = probabilité de décéder dans les n années à partir de l'âge x
    = 1 - n_p_x
    """
    return 1 - prob_survie_n_ans(df, age, n)


def loi_de_survie(df: pd.DataFrame, age_depart: int = 0) -> pd.Series:
    """
    S(t) = probabilité d'être en vie à t ans depuis age_depart
    """
    lx0 = df.loc[df["age"] == age_depart, "lx"].values[0]
    s = df[df["age"] >= age_depart]["lx"] / lx0
    s.index = df[df["age"] >= age_depart]["age"].values
    return s


# ─────────────────────────────────────────────
# 4. Comparaison inter-tables
# ─────────────────────────────────────────────

def comparer_tables(ages_cibles: list = None) -> pd.DataFrame:
    """
    Compare qx et ex sur TH00-02 vs TF00-02 vs TD88-90
    pour une liste d'âges cibles.
    """
    if ages_cibles is None:
        ages_cibles = [30, 40, 50, 60, 65, 70, 75, 80]

    resultats = []
    for table in ["TH00-02", "TF00-02", "TD88-90"]:
        df_qx = get_table_mortalite(table)
        df_full = construire_table_complete(df_qx)
        for age in ages_cibles:
            row = df_full[df_full["age"] == age]
            if not row.empty:
                resultats.append({
                    "Table": table,
                    "Age": age,
                    "qx (‰)": round(row["qx"].values[0] * 1000, 3),
                    "ex (années)": round(row["ex"].values[0], 2),
                    "lx / 100k": round(row["lx"].values[0], 0),
                })

    return pd.DataFrame(resultats)


# ─────────────────────────────────────────────
# 5. Export CSV (pour SAS)
# ─────────────────────────────────────────────

def exporter_table_csv(table: str = "TH00-02", path: str = "data/table_mortalite.csv"):
    df_qx = get_table_mortalite(table)
    df_full = construire_table_complete(df_qx)
    df_full.to_csv(path, index=False, sep=";", decimal=",")
    print(f"Table exportée : {path}")
    return df_full


if __name__ == "__main__":
    df = get_table_mortalite("TH00-02")
    df_full = construire_table_complete(df)
    print(df_full[["age", "qx", "lx", "dx", "ex"]].head(20))
    print(f"\nEspérance de vie à la naissance (TH00-02) : {df_full['ex'].iloc[0]:.2f} ans")
    print(f"Espérance de vie à 60 ans : {df_full.loc[df_full['age']==60,'ex'].values[0]:.2f} ans")
    print("\nComparaison inter-tables :")
    print(comparer_tables([30, 50, 65, 80]))

