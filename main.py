#!/usr/bin/env python3
import json
from datetime import datetime
import pandas as pd

class BondPricer:
    def __init__(self, nominal, coupon, maturity, issue_date, market_data_file=None):
        self.nominal = nominal
        self.coupon = coupon
        self.maturity = maturity
        self.issue_date = issue_date
        self.market_data_file = market_data_file
        self.results = {}

    def _discount_cashflows(self, cashflows, rates):
        """Méthode pour actualiser les flux de trésorerie."""
        discounted_value = 0
        for t, cashflow in enumerate(cashflows, start=1):
            discounted_value += cashflow / ((1 + rates[t-1]) ** t)
        return discounted_value

    def pricer_step1(self, rfr):
        """Calculer le prix d'un bond avec un taux sans risque fixe."""
        # Calcul des flux de trésorerie futurs
        cashflows = [self.coupon * self.nominal] * self.maturity
        cashflows[-1] += self.nominal  # Ajouter le remboursement du principal à la dernière période
        price = self._discount_cashflows(cashflows, [rfr] * self.maturity)
        self.results["step1"] = price
        return price

    def pricer_step2(self, rfrs):
        """Calculer le prix d'un bond avec un taux sans risque variable."""
        cashflows = [self.coupon * self.nominal] * self.maturity
        cashflows[-1] += self.nominal  # Ajouter le remboursement du principal à la dernière période
        price = self._discount_cashflows(cashflows, rfrs)
        self.results["step2"] = price
        return price

    def pricer_step3(self):
        """Calculer le prix d'un bond à partir des données du marché."""
        market_data = pd.read_csv(self.market_data_file)
        rfrs = market_data['RFR'].values[:self.maturity]
        price = self.pricer_step2(rfrs)
        self.results["step3"] = price
        return price

    def pricer_step4(self):
        """Calculer le prix d'un bond à des maturités non cotées."""
        market_data = pd.read_csv(self.market_data_file)
        rfrs = market_data['RFR'].values[:self.maturity]
        price = self.pricer_step2(rfrs)
        self.results["step4"] = price
        return price

    def pricer_step5(self):
        """Calculer le prix total et le coupon couru."""
        # Calcul du coupon couru
        current_date = datetime.now()
        days_since_last_coupon = (current_date - datetime.strptime(self.issue_date, "%d/%m/%Y")).days
        accrual = (self.coupon * self.nominal) * (days_since_last_coupon / 360)  # Base 360 jours
        market_data = pd.read_csv(self.market_data_file)
        rfrs = market_data['RFR'].values[:self.maturity]
        price = self.pricer_step2(rfrs)
        self.results["step5"] = {"npv": price, "accrual": accrual}
        return {"npv": price, "accrual": accrual}

    def save_results_to_json(self, filename="results.json"):
        """Enregistrer les résultats dans un fichier JSON."""
        with open(filename, 'w') as json_file:
            json.dump(self.results, json_file, indent=4)

    def load_results_from_json(self, filename="results.json"):
        """Lire les résultats à partir d'un fichier JSON."""
        with open(filename, 'r') as json_file:
            self.results = json.load(json_file)
        return self.results

# Initialisation des paramètres
nominal = 100
coupon = 0.04
maturity = 5
issue_date = "16/01/2023"
market_data_file = "bonds.csv"  # Assurez-vous que ce fichier est dans le bon chemin

# Créer un objet BondPricer
pricer = BondPricer(nominal, coupon, maturity, issue_date, market_data_file)

# Calculer chaque étape
step1_result = pricer.pricer_step1(0.03)  # Exemple avec un taux sans risque fixe de 3%
step2_result = pricer.pricer_step2([0.02, 0.025, 0.03, 0.035, 0.04])  # Exemple avec des taux sans risque variables
step3_result = pricer.pricer_step3()
step4_result = pricer.pricer_step4()
step5_result = pricer.pricer_step5()

# Sauvegarder les résultats dans un fichier JSON
pricer.save_results_to_json()

# Charger les résultats depuis le fichier JSON
loaded_results = pricer.load_results_from_json()
print(loaded_results)
