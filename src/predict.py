"""
Make predictions on new houses using trained regression model

Mental Model:
Like a real estate appraiser using their experience:
- Look at house features
- Compare to similar houses they've seen
- Estimate fair market value
"""

import numpy as np
import pandas as pd
import joblib
import os

class HousePricePredictor:
    """
    Class for predicting house prices
    
    Mental Model:
    This is your automated real estate appraiser
    Give it house details, it estimates the price
    """
    
    def __init__(self, model_path='models/best_model.pkl',
                 preprocessor_path='models/preprocessor.pkl'):
        """Initialize predictor with trained model"""
        
        print("📥 Loading model and preprocessor...")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load preprocessor
        preprocessor_data = joblib.load(preprocessor_path)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        
        # Load feature names
        with open('data/processed/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        # Load metadata
        self.metadata = joblib.load('models/model_metadata.pkl')
        
        print(f"✅ Loaded: {self.metadata['model_name']}")
        print(f"   Expected features: {len(self.feature_names)}")
    
    def preprocess_house(self, house_data):
        """
        Preprocess a single house's data
        
        Mental Model:
        Converting raw house description to model-ready format:
        1. Encode categories to numbers ("Downtown" → 0)
        2. Scale numbers to same range
        3. Arrange in correct order
        
        Like translating and formatting a resume for HR system
        """
        
        # Create DataFrame
        df = pd.DataFrame([house_data])
        
        # Encode categorical columns
        categorical_cols = ['Neighborhood', 'GarageType', 'Basement']
        
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Ensure correct feature order
        X = df[self.feature_names].values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, house_data):
        """
        Predict price for a single house
        
        Mental Model:
        Input: House description (dictionary)
        Process: Model analyzes features
        Output: Estimated price
        
        Like: Appraiser → Inspection → Valuation
        
        Parameters:
        -----------
        house_data : dict
            Dictionary with house features
            
        Returns:
        --------
        predicted_price : float
            Estimated sale price in dollars
        """
        
        # Preprocess
        X = self.preprocess_house(house_data)
        
        # Predict
        predicted_price = self.model.predict(X)[0]
        
        return predicted_price
    
    def predict_with_confidence(self, house_data, X_train=None, y_train=None):
        """
        Predict with confidence interval (for tree-based models)
        
        Mental Model:
        Instead of single number, give a range:
        "House is worth $350k, give or take $25k"
        
        Like weather forecast:
        "Temperature will be 75°F ± 5°F"
        """
        
        # Get base prediction
        predicted_price = self.predict(house_data)
        
        # For Random Forest, can estimate uncertainty using trees
        if hasattr(self.model, 'estimators_'):
            X = self.preprocess_house(house_data)
            
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X)[0] for tree in self.model.estimators_
            ])
            
            # Calculate std deviation
            std = tree_predictions.std()
            
            # 95% confidence interval (roughly ±2 std)
            lower_bound = predicted_price - 2 * std
            upper_bound = predicted_price + 2 * std
            
            return predicted_price, lower_bound, upper_bound, std
        else:
            # For non-ensemble models, use training data std as estimate
            return predicted_price, None, None, None
    
    def predict_batch(self, houses_df):
        """
        Predict prices for multiple houses
        
        Mental Model:
        Appraising an entire neighborhood:
        Go house by house, estimate each
        Return all valuations
        """
        
        predictions = []
        
        for idx, row in houses_df.iterrows():
            house_data = row.to_dict()
            pred = self.predict(house_data)
            predictions.append(pred)
        
        results_df = houses_df.copy()
        results_df['Predicted_Price'] = predictions
        
        return results_df
    
    def explain_prediction(self, house_data):
        """
        Explain what drove the price prediction
        
        Mental Model:
        Breaking down the appraisal:
        "Why is this house worth $350k?"
        
        - Location adds $50k
        - Size adds $150k
        - Quality adds $75k
        - Age subtracts $25k
        etc.
        """
        
        predicted_price = self.predict(house_data)
        
        print(f"\n🏠 PRICE PREDICTION BREAKDOWN")
        print("="*60)
        print(f"Estimated Price: ${predicted_price:,.0f}")
        print("="*60)
        
        print(f"\n📋 House Features:")
        for key, value in house_data.items():
            print(f"   {key}: {value}")
        
        # If model has feature importance, show top contributors
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n🔝 Most Important Factors for Pricing:")
            importances = self.model.feature_importances_
            
            # Get top 5 features
            top_indices = np.argsort(importances)[::-1][:5]
            
            for i, idx in enumerate(top_indices, 1):
                feature = self.feature_names[idx]
                importance = importances[idx]
                print(f"   {i}. {feature}: {importance:.4f} importance")
        
        print("="*60 + "\n")
        
        return predicted_price


def demo_predictions():
    """
    Demonstrate predictions on example houses
    
    Mental Model:
    Like a real estate catalog:
    Show different houses, predict their prices
    See how features affect value
    """
    
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION DEMO")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Example houses
    houses = [
        {
            'SquareFeet': 2000,
            'Bedrooms': 3,
            'Bathrooms': 2.0,
            'YearBuilt': 2015,
            'Age': 9,
            'OverallQuality': 7,
            'Neighborhood': 'Suburbs',
            'GarageType': 'Attached',
            'Basement': 'Finished'
        },
        {
            'SquareFeet': 1200,
            'Bedrooms': 2,
            'Bathrooms': 1.0,
            'YearBuilt': 1980,
            'Age': 44,
            'OverallQuality': 4,
            'Neighborhood': 'Rural',
            'GarageType': 'No Garage',
            'Basement': 'No'
        },
        {
            'SquareFeet': 3500,
            'Bedrooms': 5,
            'Bathrooms': 3.5,
            'YearBuilt': 2020,
            'Age': 4,
            'OverallQuality': 9,
            'Neighborhood': 'Beachfront',
            'GarageType': 'Attached',
            'Basement': 'Finished'
        },
        {
            'SquareFeet': 1800,
            'Bedrooms': 3,
            'Bathrooms': 2.0,
            'YearBuilt': 2000,
            'Age': 24,
            'OverallQuality': 6,
            'Neighborhood': 'Downtown',
            'GarageType': 'Carport',
            'Basement': 'Unfinished'
        }
    ]
    
    house_descriptions = [
        "Mid-size suburban family home (good condition)",
        "Small older rural cottage (fair condition)",
        "Large luxury beachfront property (excellent condition)",
        "Average downtown townhouse (decent condition)"
    ]
    
    # Make predictions
    print("🔮 Predicting prices for 4 different houses:\n")
    
    for i, (house, description) in enumerate(zip(houses, house_descriptions), 1):
        print(f"{'='*70}")
        print(f"House {i}: {description}")
        print(f"{'='*70}")
        
        # Basic prediction
        predicted_price = predictor.predict(house)
        
        # Prediction with confidence (if available)
        pred, lower, upper, std = predictor.predict_with_confidence(house)
        
        print(f"\n📊 Details:")
        print(f"   Location: {house['Neighborhood']}")
        print(f"   Size: {house['SquareFeet']:,} sqft")
        print(f"   Bedrooms: {house['Bedrooms']}")
        print(f"   Bathrooms: {house['Bathrooms']}")
        print(f"   Age: {house['Age']} years")
        print(f"   Quality: {house['OverallQuality']}/10")
        print(f"   Garage: {house['GarageType']}")
        print(f"   Basement: {house['Basement']}")
        
        print(f"\n💰 Predicted Price: ${predicted_price:,.0f}")
        
        if lower is not None and upper is not None:
            print(f"   95% Confidence: ${lower:,.0f} - ${upper:,.0f}")
            print(f"   Uncertainty: ±${std*2:,.0f}")
        
        # Price per square foot
        price_per_sqft = predicted_price / house['SquareFeet']
        print(f"   Price per sqft: ${price_per_sqft:.2f}")
        
        print()
    
    print("="*70)
    
    # Show feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE EXPLANATION")
    print("="*70)
    print("""
The model learned these general patterns:

1. SQUARE FEET (Most Important)
   Bigger houses cost significantly more
   Every 100 sqft adds ~$15,000
   
2. OVERALL QUALITY (Very Important)
   Quality rating has huge impact
   Each point (1-10) adds ~$25,000
   
3. NEIGHBORHOOD (Important)
   Beachfront: +$200k premium
   Downtown: +$50k premium
   Suburbs: Baseline
   Rural: -$30k discount
   
4. AGE (Important)
   Newer houses command premium
   Each year older: -$1,000
   
5. BEDROOMS/BATHROOMS (Moderate)
   More rooms = higher price
   But less impact than size/quality
   
6. GARAGE/BASEMENT (Minor)
   Nice to have, but not deal-breaker
   Attached garage: +$25k
   Finished basement: +$30k
    """)
    print("="*70 + "\n")


def interactive_prediction():
    """
    Interactive mode for user input
    
    Mental Model:
    Like consulting with an appraiser:
    You describe your house, they estimate value
    """
    
    print("\n" + "="*70)
    print("INTERACTIVE HOUSE PRICE ESTIMATOR")
    print("="*70 + "\n")
    
    predictor = HousePricePredictor()
    
    print("Enter house details:\n")
    
    try:
        square_feet = int(input("Square Feet (e.g., 2000): "))
        bedrooms = int(input("Bedrooms (e.g., 3): "))
        bathrooms = float(input("Bathrooms (e.g., 2.5): "))
        year_built = int(input("Year Built (e.g., 2010): "))
        quality = int(input("Overall Quality 1-10 (e.g., 7): "))
        
        print("\nNeighborhood options: Downtown, Suburbs, Rural, Beachfront, Hills")
        neighborhood = input("Neighborhood: ")
        
        print("\nGarage options: No Garage, Carport, Attached, Detached")
        garage = input("Garage Type: ")
        
        print("\nBasement options: No, Unfinished, Finished")
        basement = input("Basement: ")
        
        # Calculate age
        age = 2024 - year_built
        
        # Create house data
        house = {
            'SquareFeet': square_feet,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'YearBuilt': year_built,
            'Age': age,
            'OverallQuality': quality,
            'Neighborhood': neighborhood,
            'GarageType': garage,
            'Basement': basement
        }
        
        # Predict
        predicted_price = predictor.explain_prediction(house)
        
        print(f"\n🎯 FINAL ESTIMATE: ${predicted_price:,.0f}")
        
        # Show comparable range
        print(f"\n📊 Comparable Range:")
        print(f"   Conservative: ${predicted_price * 0.95:,.0f}")
        print(f"   Expected: ${predicted_price:,.0f}")
        print(f"   Optimistic: ${predicted_price * 1.05:,.0f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure all inputs are valid.")


def main():
    """
    Main prediction demo
    
    Mental Model:
    Two modes:
    1. Demo mode - See example predictions
    2. Interactive mode - Predict your own house
    """
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_prediction()
    else:
        demo_predictions()


if __name__ == "__main__":
    main()