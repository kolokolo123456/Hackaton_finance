import pandas as pd
import numpy as np
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

    # Calcul des flux de trésorerie
    cash_flows = np.array([nominal * coupon_rate] * maturity_years)
    cash_flows[-1] += nominal

    # Périodes de temps
    time_periods = np.arange(1, maturity_years + 1)

    # Prix actualisé
    discounted_price = np.sum(cash_flows / (1 + rfr) ** time_periods)
    return discounted_price

def days_to_years(days):
    """Convertit des jours en années en considérant 360 jours/an."""
    return days / 360

class BondCSVPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # df contient les données lues à partir de fichier csv (type pandas.DataFrame)
        self.df = None
        
    def load_data(self):
        """Charge les données du fichier CSV dans un DataFrame pandas."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Le fichier {self.file_path} a été chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du fichier: {e}")

########################################################################################################################################################################################
    
    def clean_data(self):
        """Traite les valeurs manquantes, nettoie et estime les prix manquants."""
        if self.df is None:
            print("Aucune donnée disponible pour le nettoyage.")
            return

        # Date de référence pour le calcul des maturités
        reference_date = datetime(2025, 1, 16)

        # 1. Remplir les nominals manquants avec 100
        self.df['Nominal'] = self.df['Nominal'].fillna(100)

        # 2. Remplir les coupons manquants par la moyenne
        coupon_mean = self.df['Coupon %'].mean()
        self.df['Coupon %'] = self.df['Coupon %'].fillna(coupon_mean)

        # 3. Calcul des maturités en années à partir de la date de maturité
        self.df['Maturité'] = pd.to_datetime(self.df['Maturité'], errors='coerce')
        self.df['Maturity Years'] = self.df['Maturité'].apply(
            lambda x: days_to_years((x - reference_date).days) if pd.notnull(x) else np.nan
        )
        # Supprimer les obligations avec des maturités invalides
        self.df.dropna(subset=['Maturity Years'], inplace=True)

        # 4. Estimer les prix de marché manquants avec la fonction `price_fixed_rate_bond_precise`
        rfr = 3.0  # Taux sans risque en pourcentage (ajuster selon le contexte)
        for idx, row in self.df.iterrows():
            if pd.isnull(row['Prix marché (clean)']):
                nominal = row['Nominal']
                coupon_rate = row['Coupon %']
                maturity_years = int(row['Maturity Years'])
                self.df.at[idx, 'Prix marché (clean)'] = price_fixed_rate_bond_precise(
                    nominal, coupon_rate, maturity_years, rfr
                )

        # Validation finale : supprimer les lignes restantes avec des valeurs manquantes
        self.df.dropna(subset=['Prix marché (clean)'], inplace=True)

        print("Nettoyage des données terminé.")

    def process_data(self):
        """Effectue les étapes de prétraitement robustes."""
        self.load_data()
        if self.df is not None:
            self.clean_data()
            print("Le prétraitement des données a été effectué avec robustesse.")

########################################################################################################################################################################################

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


if __name__ == "__main__":
    file_path = '~/Hackaton_finance/bonds.csv'
    preprocessor = BondCSVPreprocessor(file_path)
    preprocessor.process_data()
    preprocessor.show_table()
    preprocessor.save_cleaned_data('cleaned_bonds.csv')