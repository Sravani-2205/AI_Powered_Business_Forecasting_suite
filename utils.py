#utils.py

import logging

def simulate_sales(ad_spend, inflation_rate, base_sales):
    try:
        impact = (ad_spend * 0.05) - (inflation_rate * 0.03)
        simulated_sales = base_sales + impact
        logging.info("Simulation completed.")
        return round(simulated_sales, 2)
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        raise