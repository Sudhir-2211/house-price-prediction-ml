"""Generates realistic house price regression dataset.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import numpy as np
import pandas as pd


def generate_housing_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    sqft_living = np.random.normal(2100, 750, n_samples).clip(650, 5800)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.20, 0.40, 0.25, 0.08, 0.02], size=n_samples)
    bathrooms = (bedrooms * 0.5 + np.random.choice([0.5, 1.0, 1.5], size=n_samples)).clip(1.0, 5.0)

    location_tiers = ["Urban Prime", "Suburban Central", "Outer Suburban", "Rural Scenic"]
    location = np.random.choice(location_tiers, p=[0.25, 0.40, 0.25, 0.10], size=n_samples)

    year_built = np.random.randint(1950, 2024, size=n_samples)
    lot_size_acres = np.random.exponential(0.35, size=n_samples).clip(0.05, 4.0)
    garage_spaces = np.random.choice([0, 1, 2, 3], p=[0.10, 0.30, 0.45, 0.15], size=n_samples)
    condition_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.03, 0.12, 0.55, 0.22, 0.08], size=n_samples)
    has_pool = np.random.choice([0, 1], p=[0.82, 0.18], size=n_samples)

    # Base price calculation with realistic economic weights
    location_multipliers = {
        "Urban Prime": 1.45,
        "Suburban Central": 1.15,
        "Outer Suburban": 0.90,
        "Rural Scenic": 0.80,
    }
    loc_mult = np.array([location_multipliers[loc] for loc in location])

    # Base price per sqft ~$160-$220
    base_price = (
        sqft_living * 185.0 +
        bedrooms * 12000.0 +
        bathrooms * 18000.0 +
        (year_built - 1950) * 850.0 +
        lot_size_acres * 25000.0 +
        garage_spaces * 14000.0 +
        (condition_rating - 3) * 22000.0 +
        has_pool * 35000.0
    )

    # Apply location factor and realistic economic noise
    noise = np.random.normal(1.0, 0.07, n_samples)
    price = np.round((base_price * loc_mult * noise) / 1000.0) * 1000.0
    price = price.clip(120000.0, 1850000.0)

    df = pd.DataFrame({
        "sqft_living": np.round(sqft_living).astype(int),
        "bedrooms": bedrooms,
        "bathrooms": np.round(bathrooms, 1),
        "location_tier": location,
        "year_built": year_built,
        "lot_size_acres": np.round(lot_size_acres, 2),
        "garage_spaces": garage_spaces,
        "condition_rating": condition_rating,
        "has_pool": has_pool,
        "price": price.astype(int),
    })

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    df = generate_housing_data()
    out_path = os.path.join(data_dir, "house_prices.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} house price samples saved to {out_path}.")
