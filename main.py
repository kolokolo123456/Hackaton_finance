#!/usr/bin/env python3
import json
from datetime import datetime
import pandas as pd

def pricer_step1(nominal, coupon, maturity, rfr):
    price = 0
    for t in range(1, maturity + 1):
        # Calculer l'actualisation des coupons
        price += coupon * nominal / (1 + rfr) ** t
    # Ajouter le remboursement du principal à la fin de la maturité
    price += nominal / (1 + rfr) ** maturity
    return price

# Paramètres
nominal = 100
coupon = 0.04  # 4% par an
maturity = 5  # 5 ans
rfr = 0.03  # 3% taux sans risque fixe

# Calcul du prix de l'obligation
price = pricer_step1(nominal, coupon, maturity, rfr)
print(f"Le prix de l'obligation est : {price:.2f} €")
