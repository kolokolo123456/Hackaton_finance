from dataPreProcessing import BondCSVPreprocessor
from etape1 import price_fixed_rate_bond_precise
from etape3 import ytm
import pandas as pd

def extract_closest_maturities(file_path, target_maturity_years, reference_date):
    """
    Extrait les deux obligations ayant les maturités les plus proches d'une cible donnée.

    Args:
    - file_path (str): Chemin vers le fichier CSV contenant les données d'obligations.
    - target_maturity_years (float): Maturité cible en années.
    - reference_date (str): Date de référence sous forme de chaîne au format 'YYYY-MM-DD'.

    Returns:
    - DataFrame: Les deux obligations ayant les maturités les plus proches de la cible.
    """
    preprocessor = BondCSVPreprocessor(file_path)
    preprocessor.process_data()

    # Conversion de la colonne 'Maturité' en datetime
    preprocessor.df['Maturité'] = pd.to_datetime(preprocessor.df['Maturité'], errors='coerce')

    # Calcul des années restantes jusqu'à la maturité
    ref_date = pd.to_datetime(reference_date)
    preprocessor.df['Years to Maturity'] = (preprocessor.df['Maturité'] - ref_date).dt.days / 360

    # Calculer la différence avec la maturité cible
    preprocessor.df['Maturity Difference'] = abs(preprocessor.df['Years to Maturity'] - target_maturity_years)

    # Extraire les deux obligations ayant les maturités les plus proches
    closest_bonds = preprocessor.df.nsmallest(3, 'Maturity Difference')

    return closest_bonds


if __name__ == "__main__":

    file_path = '~/Hackaton_finance/bonds.csv'
    target_maturity_years = 7
    reference_date = '2025-01-16'

    try:
        closest_bonds = extract_closest_maturities(file_path, target_maturity_years, reference_date)
        print("Les deux obligations ayant les maturités les plus proches sont :")
        print(closest_bonds[['ISIN', 'Name', 'Years to Maturity', 'Coupon %', 'Prix marché (clean)']])

        # Calculer les YTM pour les deux obligations
        r_values = []
        for _, row in closest_bonds.iterrows():
            price = float(row['Prix marché (clean)'])
            nominal = float(row['Nominal'])
            coupon_rate = float(row['Coupon %'])
            maturity_years = float(row['Years to Maturity'])
            annual_coupon = nominal * (coupon_rate / 100)
            r = ytm(price, nominal, annual_coupon, int(maturity_years))
            if (r > 0):
                r_values.append(r)
            print(f"YTM pour l'obligation {row['ISIN']} : {r:.6%}")

        # Interpolation pour trouver le rendement correspondant à 7 ans
        ytm_1, ytm_2 = r_values
        t1, t2 = closest_bonds['Years to Maturity'].iloc[0], closest_bonds['Years to Maturity'].iloc[1]
        interpolated_r = ytm_1 + (target_maturity_years - t1) * (ytm_2 - ytm_1) / (t2 - t1)
        print(f"Rendement interpolé pour une maturité de {target_maturity_years} ans : {interpolated_r:.6%}")

        # Calculer le prix de l'obligation avec le rendement interpolé
        nominal = 100
        coupon_rate = 5
        interpolated_price = price_fixed_rate_bond_precise(nominal, coupon_rate, target_maturity_years, interpolated_r * 100)
        print(f"Prix estimé de l'obligation avec une maturité de {target_maturity_years} ans : {interpolated_price:.10f}")

    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
