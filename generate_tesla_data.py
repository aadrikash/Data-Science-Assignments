import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Define columns and categories
years = list(range(2015, 2026))
months = list(range(1, 13))
regions = ['North America', 'Europe', 'Asia-Pacific', 'Rest of World']
models = ['Model S', 'Model X', 'Model 3', 'Model Y', 'Cybertruck']

# Historical annual target deliveries for Tesla (approximate actuals)
annual_targets = {
    2015: 50580,
    2016: 76230,
    2017: 103020,
    2018: 245240,
    2019: 367500,
    2020: 499550,
    2021: 936172,
    2022: 1313851,
    2023: 1808581,
    2024: 1789226,
    2025: 1636129  # Projected/Estimated for 2025
}

# Regional distribution weights
region_weights = {
    'North America': 0.40,
    'Europe': 0.28,
    'Asia-Pacific': 0.26,
    'Rest of World': 0.06
}

# Model characteristics
model_specs = {
    'Model S': {'start_year': 2015, 'base_price': 90000, 'battery': 95.0, 'base_range': 500, 'share': 0.12},
    'Model X': {'start_year': 2016, 'base_price': 100000, 'battery': 100.0, 'base_range': 450, 'share': 0.10},
    'Model 3': {'start_year': 2017, 'base_price': 42000, 'battery': 70.0, 'base_range': 430, 'share': 0.35},
    'Model Y': {'start_year': 2020, 'base_price': 52000, 'battery': 78.0, 'base_range': 420, 'share': 0.40},
    'Cybertruck': {'start_year': 2023, 'base_price': 75000, 'battery': 120.0, 'base_range': 480, 'share': 0.03}
}

data = []

for year in years:
    target_deliveries = annual_targets[year]
    for month in months:
        # Determine seasonal factor (quarterly push in months 3, 6, 9, 12)
        if month in [3, 6, 9, 12]:
            season_factor = 1.35 + np.random.normal(0, 0.05)  # Eoq push
        elif month in [1, 4, 7, 10]:
            season_factor = 0.70 + np.random.normal(0, 0.05)  # Start of quarter lag
        else:
            season_factor = 0.95 + np.random.normal(0, 0.04)  # Mid quarter
            
        # Monthly base deliveries before splitting by model/region
        monthly_base = (target_deliveries / 12) * season_factor
        
        for model in models:
            specs = model_specs[model]
            # Check if model was released
            if year < specs['start_year']:
                continue
            if year == specs['start_year'] and month < 6: # Mid-year ramp up
                continue
                
            # Model pricing fluctuations (inflation, price cuts in 2023-2024)
            price_mult = 1.0
            if year in [2021, 2022]:
                price_mult = 1.08  # Inflation/supply chain peaks
            elif year in [2023, 2024, 2025]:
                price_mult = 0.88  # Tesla price cuts
                
            avg_price = specs['base_price'] * price_mult * np.random.uniform(0.97, 1.03)
            
            # Efficiency improvements over the years (increase range slightly)
            range_trend = specs['base_range'] * (1 + (year - specs['start_year']) * 0.015)
            avg_range = range_trend * np.random.uniform(0.98, 1.02)
            
            # Battery size (fairly constant, tiny upgrades)
            battery_cap = specs['battery'] * np.random.uniform(0.99, 1.01)
            
            for region in regions:
                r_weight = region_weights[region]
                
                # Model share changes dynamically over the years (Model Y becomes dominant)
                if year < 2017:
                    # Only S and X active
                    m_weight = 0.6 if model == 'Model S' else 0.4
                elif year < 2020:
                    # S, X, 3 active
                    shares = {'Model S': 0.15, 'Model X': 0.10, 'Model 3': 0.75}
                    m_weight = shares.get(model, 0.0)
                else:
                    # All models potentially active
                    # Adjust shares depending on what's active
                    active_shares = {}
                    if year >= 2023:
                        active_shares = {'Model S': 0.04, 'Model X': 0.04, 'Model 3': 0.30, 'Model Y': 0.57, 'Cybertruck': 0.05}
                    else:
                        active_shares = {'Model S': 0.05, 'Model X': 0.05, 'Model 3': 0.35, 'Model Y': 0.55}
                    m_weight = active_shares.get(model, 0.0)
                
                # Compute base production units
                prod_base = monthly_base * r_weight * m_weight
                production_units = int(max(10, np.round(prod_base * np.random.uniform(0.92, 1.08))))
                
                # Deliveries lag or lead production (logistics, inventory)
                delivery_factor = np.random.normal(0.98, 0.05)
                # Extra EOQ delivery push
                if month in [3, 6, 9, 12]:
                    delivery_factor += np.random.uniform(0.01, 0.05)
                    
                estimated_deliveries = int(max(5, np.round(production_units * delivery_factor)))
                
                # CO2 saved in tons (approx. 0.12 tons of CO2 saved per 1000 km driven by EV vs ICE)
                # We can assume average annual driving or estimate saved tons per vehicle per year
                # Let's make CO2 saved a function of deliveries, battery capacity and range
                co2_saved = estimated_deliveries * (avg_range * 0.00018) * np.random.uniform(0.95, 1.05)
                
                data.append({
                    'Year': year,
                    'Month': month,
                    'Region': region,
                    'Model': model,
                    'Production_Units': production_units,
                    'Avg_Price_USD': np.round(avg_price, 2),
                    'Battery_Capacity_kWh': np.round(battery_cap, 1),
                    'Range_km': np.round(avg_range, 1),
                    'CO2_Saved_tons': np.round(co2_saved, 2),
                    'Estimated_Deliveries': estimated_deliveries
                })

df = pd.DataFrame(data)

# Sort by Date logically
df = df.sort_values(by=['Year', 'Month', 'Region', 'Model']).reset_index(drop=True)

# Write to CSV
output_path = 'tesla_delivery_data.csv'
df.to_csv(output_path, index=False)
print(f"Dataset generated successfully! Shape: {df.shape}")
print(df.head())
