import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

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

