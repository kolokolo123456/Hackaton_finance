from datetime import datetime

def calculate_accrued_coupon(nominal, coupon_rate, last_payment_date, current_date):
    """
    Calcule le coupon couru.
    """
    # Calcul du coupon annuel
    annual_coupon = nominal * coupon_rate / 100
    
    # Nombre de jours écoulés depuis le dernier paiement
    days_elapsed = (current_date - last_payment_date).days
    
    return annual_coupon * (days_elapsed / 360)

def calculate_bond_price(nominal, coupon_rate, maturity, rate, current_date, issue_date):
    """
    Calcule le prix de l'obligation et le prix total incluant le coupon couru.
    """
    # Flux de trésorerie futurs
    annual_coupon = nominal * coupon_rate / 100
    future_cashflows = [
        annual_coupon / (1 + rate) ** t for t in range(1, maturity)
    ]
    final_cashflow = (annual_coupon + nominal) / (1 + rate) ** maturity
    
    bond_price = sum(future_cashflows) + final_cashflow
    
    # Dernier paiement de coupon
    last_payment_date = datetime(current_date.year - 1, issue_date.month, issue_date.day)
    
    # Calcul du coupon couru
    accrued_coupon = calculate_accrued_coupon(nominal, coupon_rate, last_payment_date, current_date)
    
    # Prix total
    total_price = bond_price + accrued_coupon
    
    return bond_price, accrued_coupon, total_price

# Données fournies
nominal = 100
coupon_rate = 4
maturity = 5
rate = 0.02349654472810973  # taux sans risque calculé en étape 3
issue_date = datetime(2023, 7, 16)
current_date = datetime(2025, 1, 16)

bond_price, accrued_coupon, total_price = calculate_bond_price(nominal, coupon_rate, maturity, rate, current_date, issue_date)

# Résultats
print(f"Prix de l'obligation (flux actualisés) : {bond_price:.2f} €")
print(f"Coupon couru : {accrued_coupon:.2f} €")
print(f"Prix total de l'obligation : {total_price:.2f} €")
