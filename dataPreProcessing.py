import pandas as pd
import numpy as np
from scipy.optimize import newton
from tabulate import tabulate
from datetime import datetime

def price_fixed_rate_bond_precise(nominal, coupon_rate, maturity_years, rfr):
    """
    Calcule le prix d'une obligation avec un taux sans risque fixe en utilisant une actualisation précise.

    Args:
    - nominal : float, valeur nominale de l'obligation.
    - coupon_rate : float, taux de coupon annuel (en pourcentage).
    - maturity_years : int, nombre d'années jusqu'à la maturité.
    - rfr : float, taux sans risque (en pourcentage).

    Returns:
    - float, prix actualisé de l'obligation.
    """

    coupon_rate /= 100
    rfr /= 100
    cash_flows = np.array([nominal * coupon_rate] * maturity_years)
    cash_flows[-1] += nominal
    time_periods = np.arange(1, maturity_years + 1)
    discounted_price = np.sum(cash_flows / (1 + rfr) ** time_periods)
    return discounted_price

def days_to_years(days):
    """Convertit des jours en années en considérant 360 jours/an."""
    return days / 360

class BondCSVPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Charge les données du fichier CSV dans un DataFrame pandas."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Le fichier {self.file_path} a été chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du fichier: {e}")

    
    def clean_data(self):
        """Traite les valeurs manquantes, nettoie et estime les prix manquants."""
        if self.df is None:
            print("Aucune donnée disponible pour le nettoyage.")
            return

        reference_date = datetime(2025, 1, 16)

        self.df['Nominal'] = self.df['Nominal'].fillna(100)
        coupon_mean = self.df['Coupon %'].mean()
        self.df['Coupon %'] = self.df['Coupon %'].fillna(coupon_mean)
        self.df['Maturité'] = pd.to_datetime(self.df['Maturité'], errors='coerce')
        self.df['Maturity Years'] = self.df['Maturité'].apply(
            lambda x: days_to_years((x - reference_date).days) if pd.notnull(x) else np.nan
        )
        self.df.dropna(subset=['Maturity Years'], inplace=True)

        rfr = 3.0  
        for idx, row in self.df.iterrows():
            if pd.isnull(row['Prix marché (clean)']):
                nominal = row['Nominal']
                coupon_rate = row['Coupon %']
                maturity_years = int(row['Maturity Years'])
                self.df.at[idx, 'Prix marché (clean)'] = price_fixed_rate_bond_precise(
                    nominal, coupon_rate, maturity_years, rfr
                )

        self.df.dropna(subset=['Prix marché (clean)'], inplace=True)

        print("Nettoyage des données terminé.")

    def process_data(self):
        """Effectue les étapes de prétraitement robustes."""
        self.load_data()
        if self.df is not None:
            self.clean_data()
            print("Le prétraitement des données a été effectué avec robustesse.")


    def save_cleaned_data(self, output_path):
        """Sauvegarde le DataFrame nettoyé dans un fichier CSV."""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"Les données nettoyées ont été sauvegardées dans {output_path}.")
        else:
            print("Aucune donnée à sauvegarder.")

    def show_summary(self):
        """Affiche un résumé des données."""
        if self.df is not None:
            print(self.df.info())
            print(self.df.head())
        else:
            print("Aucune donnée chargée.")

    def show_table(self):
        """Affiche les données sous forme de tableau."""
        if self.df is not None:
            print(tabulate(self.df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print("Aucune donnée chargée pour afficher le tableau.")


def solve_for_r(price, nominal, coupon_rate, maturity_years):

    coupon = nominal * (coupon_rate / 100)
    
    def f(r):
        cash_flows = np.array([coupon] * maturity_years)
        cash_flows[-1] += nominal  # Ajouter le nominal au dernier flux
        time_periods = np.arange(1, maturity_years + 1)
        return np.sum(cash_flows / (1 + r) ** time_periods) - price
    
    r_solution = newton(f, x0=0.03, tol=1e-6, maxiter=100)
    return np.round(r_solution, 6)


if __name__ == "__main__":
    file_path = '~/Hackaton_finance/bonds.csv'
    
    preprocessor = BondCSVPreprocessor(file_path)
    preprocessor.process_data()
    
    r_values = []
    
    for idx, row in preprocessor.df.iterrows():
        price = row['Prix marché (clean)']
        nominal = row['Nominal']
        coupon_rate = row['Coupon %']
        maturity_years = int(row['Maturity Years'])
        
        r = solve_for_r(price, nominal, coupon_rate, maturity_years)
        
        if not np.isnan(r):
            r_values.append(r)
    
    if r_values:
        average_r = np.mean(r_values)
        print(f"\nLa moyenne des taux sans risque calculés est : {average_r:.6%}")
    else:
        print("\nAucun taux sans risque valide n'a été calculé.")
    
    preprocessor.save_cleaned_data('cleaned_bonds.csv')
