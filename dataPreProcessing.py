import pandas as pd
import numpy as np
from scipy.optimize import newton
from datetime import datetime
from tabulate import tabulate

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
        """Supprime les lignes incomplètes, les données expirées et nettoie les données."""
        if self.df is None:
            print("Aucune donnée disponible pour le nettoyage.")
            return

        # Date de référence
        reference_date = datetime(2025, 1, 16)

        # 1. Suppression des lignes contenant des valeurs manquantes
        self.df.dropna(inplace=True)

        # 2. Conversion de la colonne 'Maturité' en type datetime
        self.df['Maturité'] = pd.to_datetime(self.df['Maturité'], errors='coerce')

        # 3. Suppression des lignes avec des dates de maturité expirées
        self.df = self.df[self.df['Maturité'] >= reference_date]

        # 4. Recalcul des années de maturité
        self.df['Maturity Years'] = self.df['Maturité'].apply(
            lambda x: max(0, days_to_years((x - reference_date).days)) if pd.notnull(x) else np.nan
        )

        # 5. Suppression des lignes avec des maturités invalides ou manquantes
        self.df.dropna(subset=['Maturity Years'], inplace=True)

        print(f"Nettoyage des données terminé. Nombre de lignes restantes : {len(self.df)}")

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
    file_path = '~/Hackaton_finance/bonds.csv'  # Chemin vers le fichier CSV

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

        # Supprimer les lignes avec des maturités invalides
        preprocessor.df.dropna(subset=['Maturity Years'], inplace=True)

        # Affichage des données nettoyées
        print("\n--- Tableau des données nettoyées ---")
        preprocessor.show_table()

        # Étape 4 : Calcul des taux sans risque (r)
        print("\n--- Étape 4 : Calcul des taux sans risque (r) ---")
        # Filtrer les lignes valides
        preprocessor.df = preprocessor.df[
            (preprocessor.df['Maturity Years'] > 0) & 
            (preprocessor.df['Prix marché (clean)'] > 0) & 
            (preprocessor.df['Nominal'] > 0)
        ]

        if preprocessor.df.empty:
            print("Aucune donnée valide pour le calcul des taux sans risque.")
        else:
            print(f"Nombre de lignes valides pour le calcul : {len(preprocessor.df)}")

            r_values = []

            for idx, row in preprocessor.df.iterrows():
                try:
                    # Extraction des données nécessaires
                    price = float(row['Prix marché (clean)'])
                    nominal = float(row['Nominal'])
                    coupon_rate = float(row['Coupon %'])
                    maturity_years = int(row['Maturity Years'])

                    # Calcul du taux sans risque
                    r = solve_for_r(price, nominal, coupon_rate, maturity_years)

                    # Ajouter le taux sans risque calculé
                    r_values.append(r)
                    print(f"Ligne {idx}: r = {r:.6%}")

                except Exception as e:
                    print(f"Ligne {idx}: Échec du calcul de r - {e}")

            # Étape 5 : Calcul de la moyenne des r
            print("\n--- Étape 5 : Calcul de la moyenne des taux sans risque ---")
            if r_values:
                average_r = np.mean(r_values)
                print(f"La moyenne des taux sans risque calculés est : {average_r:.6%}")
            else:
                print("\nAucun taux sans risque valide n'a été calculé.")

        # Étape 6 : Sauvegarde des données nettoyées
        print("\n--- Étape 6 : Sauvegarde des données nettoyées ---")
        preprocessor.save_cleaned_data('cleaned_bonds.csv')

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

