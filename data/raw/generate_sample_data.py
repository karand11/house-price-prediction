"""
Generate sample house price data for regression project
"""

import pandas as pd
import numpy as np

def generate_house_data(n_samples=1460):
    """
    Generate realistic house price data
    
    Mental Model:
    We're creating fake house data that mimics real patterns:
    - Bigger houses cost more
    - Better quality costs more
    - Newer houses cost more
    - Better neighborhoods cost more
    + some random variation (life isn't perfectly predictable!)
    """
    
    print(f"🏠 Generating {n_samples} sample houses...")
    
    np.random.seed(42)
    
    # Basic features (normally distributed)
    square_feet = np.random.normal(1500, 500, n_samples).astype(int)
    square_feet = np.clip(square_feet, 500, 5000)  # Realistic range
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, 
                                p=[0.05, 0.15, 0.35, 0.35, 0.10])
    
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], size=n_samples,
                                 p=[0.10, 0.15, 0.35, 0.25, 0.15])
    
    # Age (year built)
    current_year = 2024
    year_built = np.random.randint(1950, 2024, n_samples)
    age = current_year - year_built
    
    # Quality rating (1-10)
    overall_quality = np.random.choice(range(1, 11), size=n_samples,
                                      p=[0.02, 0.03, 0.05, 0.10, 0.15, 
                                         0.20, 0.20, 0.15, 0.07, 0.03])
    
    # Categorical features
    neighborhoods = np.random.choice(
        ['Downtown', 'Suburbs', 'Rural', 'Beachfront', 'Hills'],
        size=n_samples,
        p=[0.25, 0.40, 0.20, 0.10, 0.05]
    )
    
    garage_type = np.random.choice(
        ['No Garage', 'Carport', 'Attached', 'Detached'],
        size=n_samples,
        p=[0.10, 0.15, 0.60, 0.15]
    )
    
    basement = np.random.choice(['No', 'Unfinished', 'Finished'],
                               size=n_samples,
                               p=[0.20, 0.30, 0.50])
    
    # Calculate price based on features
    # Mental Model: Price = Base + (factors that increase value)
    
    base_price = 100000
    
    price = (
        base_price
        + (square_feet * 150)                    # $150 per sqft
        + (bedrooms * 20000)                     # $20k per bedroom
        + (bathrooms * 15000)                    # $15k per bathroom
        + (overall_quality * 25000)              # $25k per quality point
        - (age * 1000)                           # -$1k per year of age
    )
    
    # Neighborhood bonus
    neighborhood_bonus = {
        'Beachfront': 200000,
        'Hills': 150000,
        'Downtown': 50000,
        'Suburbs': 0,
        'Rural': -30000
    }
    price += [neighborhood_bonus[n] for n in neighborhoods]
    
    # Garage bonus
    garage_bonus = {
        'Detached': 20000,
        'Attached': 25000,
        'Carport': 5000,
        'No Garage': 0
    }
    price += [garage_bonus[g] for g in garage_type]
    
    # Basement bonus
    basement_bonus = {
        'Finished': 30000,
        'Unfinished': 10000,
        'No': 0
    }
    price += [basement_bonus[b] for b in basement]
    
    # Add random noise (life isn't perfectly predictable)
    noise = np.random.normal(0, 25000, n_samples)
    price += noise
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'SquareFeet': square_feet,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'YearBuilt': year_built,
        'Age': age,
        'OverallQuality': overall_quality,
        'Neighborhood': neighborhoods,
        'GarageType': garage_type,
        'Basement': basement,
        'SalePrice': price.astype(int)
    })
    
    print(f"✅ Generated {len(df)} houses")
    print(f"   Price range: ${df['SalePrice'].min():,} - ${df['SalePrice'].max():,}")
    print(f"   Average price: ${df['SalePrice'].mean():,.0f}")
    
    return df

def main():
    """Generate and save data"""
    import os
    
    # Create directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate data
    df = generate_house_data(1460)
    
    # Save
    df.to_csv('data/raw/house_data.csv', index=False)
    print(f"\n✅ Data saved to data/raw/house_data.csv")
    
    # Show sample
    print("\n📊 Sample data:")
    print(df.head())
    
    print("\n📈 Statistics:")
    print(df.describe())


if __name__ == "__main__":
        main()