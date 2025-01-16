from datetime import datetime

def calculate_accrued_coupon(nominal, coupon_rate, last_payment_date, current_date):
    """
    Calcule le coupon couru.
    """
    # Calcul du coupon annuel
    annual_coupon = nominal * coupon_rate / 100
    
    # Nombre de jours écoulés depuis le dernier paiement
    days_elapsed = (current_date - last_payment_date).days
    
    # Nombre total de jours dans la période (360 par convention)
    total_days = 360  # Conformément aux conventions
    accrued_coupon = annual_coupon * (days_elapsed / total_days)
    
    return accrued_coupon

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
coupon_rate = 4  # en pourcentage
maturity = 5  # en années
rate = 0.03  # taux sans risque (3%)
issue_date = datetime(2023, 7, 16)
current_date = datetime(2025, 1, 16)

# Calcul
bond_price, accrued_coupon, total_price = calculate_bond_price(
    nominal, coupon_rate, maturity, rate, current_date, issue_date
)

# Résultats
print(f"Prix de l'obligation (flux actualisés) : {bond_price:.2f} €")
print(f"Coupon couru : {accrued_coupon:.2f} €")
print(f"Prix total de l'obligation : {total_price:.2f} €")
