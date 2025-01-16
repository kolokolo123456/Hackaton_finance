#!/usr/bin/env python3
import numpy as np

def price_variable_rate_bond(nominal, coupon_rate, maturity_years, rfr_schedule):
    """
    Calcule le prix d'une obligation avec un taux sans risque variable.
    
    Args:
    - nominal : float, valeur nominale de l'obligation.
    - coupon_rate : float, taux de coupon annuel (en pourcentage).
    - maturity_years : int, nombre d'années jusqu'à la maturité.
    - rfr_schedule : list of float, taux sans risque pour chaque année (en pourcentage).
    
    Returns:
    - float, prix actualisé de l'obligation.
    """
    # Convertir les taux en décimaux
    coupon_rate /= 100
    rfr_schedule = [r / 100 for r in rfr_schedule]
    
    # Calcul des flux de trésorerie : coupon chaque année + remboursement final du nominal
    cash_flows = np.array([nominal * coupon_rate] * maturity_years)
    cash_flows[-1] += nominal  # Ajouter le nominal au dernier flux
    
    # Calcul du prix actualisé en utilisant les taux variables
    discounted_price = sum(cash_flows[t] / ((1 + rfr_schedule[t]) ** (t + 1)) for t in range(maturity_years))
    
    return discounted_price



# Exécution de la fonction
if __name__ == "__main__":
    nominal = 100
    coupon_rate = 4 
    maturity_years = 5
    rfr_schedule = [2, 2.5, 3, 3.5, 4]  

    price_etape2 = price_variable_rate_bond(nominal, coupon_rate, maturity_years, rfr_schedule)
    print(f"Prix actualisé de l'obligation avec taux variable : {price_etape2:.10f} €")