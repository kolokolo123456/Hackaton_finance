import numpy as np
from scipy.optimize import newton

def solve_for_r(price, nominal, coupon_rate, maturity_years):
    """
    Résout l'équation du prix de l'obligation pour trouver le taux sans risque (r).
    
    Args:
    - price : float, prix de marché de l'obligation.
    - nominal : float, valeur nominale de l'obligation.
    - coupon_rate : float, taux de coupon annuel (en pourcentage).
    - maturity_years : int, nombre d'années jusqu'à la maturité.
    
    Returns:
    - float, taux sans risque (r) résolu.
    """
    coupon = nominal * (coupon_rate / 100)
    
    def f(r):
        cash_flows = np.array([coupon] * maturity_years)
        cash_flows[-1] += nominal  # Ajouter le nominal au dernier flux
        time_periods = np.arange(1, maturity_years + 1)
        return np.sum(cash_flows / (1 + r) ** time_periods) - price
    
    r_solution = newton(f, x0=0.03, tol=1e-6, maxiter=100)
    return np.round(r_solution, 6)


