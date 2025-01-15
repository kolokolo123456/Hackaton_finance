#!/usr/bin/env python3
import json
from datetime import datetime
import pandas as pd

def pricer_step1(nominal, coupon, maturity, rfr):
    price = 0
    for t in range(1, maturity + 1):
        price += coupon * nominal / (1 + rfr) ** t
    price += nominal / (1 + rfr) ** maturity
    # Arrondi à 4 décimales avant de retourner le prix final
    return round(price, 4)


# Paramètres
nominal = 100
coupon = 0.04  # 4% par an
maturity = 5  # 5 ans
rfr = 0.03  # 3% taux sans risque fixe

# Calcul du prix de l'obligation
price = pricer_step1(nominal, coupon, maturity, rfr)
print(f"Le prix de l'obligation est : {price:.2f} €")

https://www.omnicalculator.com/fr/finance/calculateur-prix-obligations?utm_source=chatgpt.com