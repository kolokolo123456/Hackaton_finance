from etape1 import price_fixed_rate_bond_precise
from dataPreProcessing import  BondCSVPreprocessor
from statsmodels.robust.scale import huber

import pandas as pd
import numpy as np
from datetime import datetime

def days_to_years(days):
    """Convertit des jours en années en considérant 360 jours/an."""
    return days / 360

def ytm(P, N, C, T, tol=1e-9, max_iter=10000):
    r = C / P  # Initial guess
    for _ in range(max_iter):
        f_r = P - sum(C * N / (1 + r)**t for t in range(1, T+1)) - N / (1 + r)**T
        f_prime_r = sum(t * C * N / (1 + r)**(t+1) for t in range(1, T+1)) + T * N / (1 + r)**(T+1)
        r_new = r - f_r / f_prime_r
        if abs(r_new - r) < tol:
            return r_new
        r = r_new

def estimate_rfr(ytm_list):
    # On utilise l'estimateur de huber pour gérer les valeurs extrêmes de r
    rfr, _ = huber(ytm_list * 100)
    return rfr


if __name__ == "__main__":
    file_path = '~/Hackaton_finance/bonds.csv'

    # Initialisation du préprocesseur
    preprocessor = BondCSVPreprocessor(file_path)

    try:
        # Étape 1 : Prétraitement des données
        print("\n--- Étape 1 : Prétraitement des données ---")
        preprocessor.process_data()

        # Calcul de Maturity Years
        reference_date = datetime(2025, 1, 16)  # Date de référence
        preprocessor.df['Maturity Years'] = preprocessor.df['Maturité'].apply(
            lambda x: max(0, days_to_years((pd.to_datetime(x) - reference_date).days))
            if pd.notnull(x) else np.nan
        )

        # Affichage du tableau nettoyé
        # preprocessor.show_table()

        # Étape 4 : Calcul des taux sans risque (r)
        print("\n--- Étape 4 : Calcul des taux sans risque (r) ---")
        r_values = []

        for idx, row in preprocessor.df.iterrows():
            # Extraction des données nécessaires
            price = float(row['Prix marché (clean)'])
            nominal = float(row['Nominal'])
            coupon_rate = float(row['Coupon %'])
            maturity_years = int(row['Maturity Years'])

            # Calcul du taux sans risque
            r = ytm(price, nominal, coupon_rate, maturity_years)

            # Ajouter le taux sans risque calculé
            r_values.append(r)
            print(f"Ligne {idx}: r = {r:.6%}")

        # Étape 5 : Calcul de la moyenne des r
        print("\n--- Étape 5 : Calcul de la moyenne des taux sans risque ---")
        average_r = estimate_rfr(r_values)
        print(f"La moyenne des taux sans risque calculés est : {average_r:.6%}")

        # Étape 6 : Estimation du prix du bond
        bond_price = price_fixed_rate_bond_precise(100, 4, 5, average_r)
        print(f"Le prix du bond estimé par l'étape 3 : ", bond_price)

        # Étape 7 : Sauvegarde des données nettoyées
        print("\n--- Étape 6 : Sauvegarde des données nettoyées ---")
        preprocessor.save_cleaned_data('cleaned_bonds.csv')

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

