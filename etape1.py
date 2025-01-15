#!/usr/bin/env python3
import numpy as np


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


if __name__ == "__main__":
    nominal = 100
    coupon_rate = 4 
    maturity_years = 5
    rfr = 3 
    
    price = price_fixed_rate_bond_precise(nominal, coupon_rate, maturity_years, rfr)
    print(f"Prix actualisé de l'obligation avec taux fixe : {price:.6f} €")
