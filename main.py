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
    # Convertir les taux en décimaux
    coupon_rate /= 100
    rfr /= 100
    
    # Calcul des flux de trésorerie : coupon chaque année + remboursement final du nominal
    cash_flows = np.array([nominal * coupon_rate] * maturity_years)
    cash_flows[-1] += nominal  # Ajouter le nominal au dernier flux
    
    # Calcul du prix actualisé avec actualisation discrète
    time_periods = np.arange(1, maturity_years + 1)
    discounted_price = np.sum(cash_flows / (1 + rfr) ** time_periods)
    
    return discounted_price

# Exécution de la fonction
if __name__ == "__main__":
    nominal = 100
    coupon_rate = 5  # En pourcentage
    maturity_years = 5
    rfr = 3  # En pourcentage
    
    price = price_fixed_rate_bond_precise(nominal, coupon_rate, maturity_years, rfr)
    print(f"Prix actualisé de l'obligation : {price:.6f} €")
