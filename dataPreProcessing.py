import pandas as pd
import numpy as np
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
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        after_dropna_rows = len(self.df)
        removed_due_to_na = initial_rows - after_dropna_rows

        # 2. Conversion de la colonne 'Maturité' en type datetime
        self.df['Maturité'] = pd.to_datetime(self.df['Maturité'], errors='coerce')

        # 3. Suppression des lignes avec des dates de maturité expirées
        self.df = self.df[self.df['Maturité'] >= reference_date]
        final_rows = len(self.df)
        removed_due_to_expired_dates = after_dropna_rows - final_rows

        # 4. Résumé des suppressions
        total_removed = removed_due_to_na + removed_due_to_expired_dates
        print(f"{total_removed} lignes supprimées au total.")
        print(f"- {removed_due_to_na} lignes supprimées à cause de valeurs manquantes.")
        print(f"- {removed_due_to_expired_dates} lignes supprimées à cause de dates de maturité expirées.")

        # Validation finale
        if self.df.isnull().any().any():
            raise ValueError("Certaines valeurs manquantes persistent après le nettoyage.")

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

    def show_table(self):
        """Affiche les données sous forme de tableau."""
        if self.df is not None:
            print(tabulate(self.df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print("Aucune donnée chargée pour afficher le tableau.")


def ytm(P, N, C, T, tol=1e-9, max_iter=10000):
    r = C / P  # Initial guess
    for _ in range(max_iter):
        f_r = P - sum(C * N / (1 + r)**t for t in range(1, T+1)) - N / (1 + r)**T
        f_prime_r = sum(t * C * N / (1 + r)**(t+1) for t in range(1, T+1)) + T * N / (1 + r)**(T+1)
        r_new = r - f_r / f_prime_r
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    raise ValueError("Convergence not achieved within max iterations")

def estimate_rfr(ytm_list, method="weighted", maturities=None):
    # Vérification des données
    if method in ["weighted", "long_term"] and maturities is None:
        raise ValueError("Les maturités doivent être fournies pour les méthodes 'weighted' et 'long_term'.")
    if method == "weighted" and len(ytm_list) != len(maturities):
        raise ValueError("Les longueurs de 'ytm_list' et 'maturities' doivent être identiques.")

    # Estimation selon la méthode choisie
    if method == "weighted":
        rfr = np.sum(np.array(ytm_list) * np.array(maturities)) / np.sum(maturities)
    elif method == "long_term":
        long_term_indices = [i for i, t in enumerate(maturities) if t > 10]
        long_term_ytms = [ytm_list[i] for i in long_term_indices]
        rfr = np.mean(long_term_ytms) if long_term_ytms else None
    elif method == "minimal":
        rfr = min(ytm_list)
    else:
        raise ValueError(f"Méthode '{method}' non reconnue.")

    return rfr


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

        # Supprimer les lignes avec une maturité invalide
        preprocessor.df.dropna(subset=['Maturity Years'], inplace=True)

        # Affichage du tableau nettoyé
        preprocessor.show_table()

        # Étape 4 : Calcul des taux sans risque (r)
        print("\n--- Étape 4 : Calcul des taux sans risque (r) ---")
        r_values = []

        for idx, row in preprocessor.df.iterrows():
            try:
                # Extraction des données nécessaires
                price = float(row['Prix marché (clean)'])
                nominal = float(row['Nominal'])
                coupon_rate = float(row['Coupon %'])
                maturity_years = int(row['Maturity Years'])

                # Vérifications de cohérence des données
                if price <= 0 or nominal <= 0 or coupon_rate < 0 or maturity_years <= 0:
                    raise ValueError("Données incohérentes : les valeurs doivent être positives.")

                # Calcul du taux sans risque
                r = ytm(price, nominal, coupon_rate, maturity_years)

                # Ajouter le taux sans risque calculé
                r_values.append(r)
                print(f"Ligne {idx}: r = {r:.6%}")

            except ValueError as ve:
                print(f"Ligne {idx}: Erreur de validation des données - {ve}")
            except Exception as e:
                print(f"Ligne {idx}: Échec du calcul de r - {e}")

        # Étape 5 : Calcul de la moyenne des r
        print("\n--- Étape 5 : Calcul de la moyenne des taux sans risque ---")
        if r_values:
            average_r = estimate_rfr(r_values, "weighted", preprocessor.df['Maturity Years'])
            print(f"La moyenne des taux sans risque calculés est : {average_r:.6%}")
        else:
            print("\nAucun taux sans risque valide n'a été calculé.")

        # Étape 6 : Sauvegarde des données nettoyées
        print("\n--- Étape 6 : Sauvegarde des données nettoyées ---")
        preprocessor.save_cleaned_data('cleaned_bonds.csv')

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

