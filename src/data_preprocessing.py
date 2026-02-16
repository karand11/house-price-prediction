"""
Data preprocessing for house price prediction (REGRESSION)

Mental Model:
This is like preparing ingredients before cooking:
1. Wash vegetables (clean data)
2. Chop to uniform size (encode categories)
3. Measure portions (scale numbers)
4. Separate for cooking (train/test split)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class HousePricePreprocessor:
    """
    Preprocess house price data for regression
    
    Mental Model:
    Think of this as a factory assembly line:
    Raw data → Clean → Encode → Scale → Split → Ready!
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath='data/raw/house_data.csv'):
        """Load raw data"""
        print(f"📥 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} houses with {df.shape[1]} features")
        return df
    
    def clean_data(self, df):
        """
        Clean and validate data
        
        Mental Model:
        Like a quality inspector checking products:
        - Remove broken items (extreme outliers)
        - Fix minor issues (missing values)
        - Validate all items meet standards
        """
        print("🧹 Cleaning data...")
        
        df_clean = df.copy()
        
        # Check for missing values
        missing = df_clean.isnull().sum()
        if missing.sum() > 0:
            print(f"   Found {missing.sum()} missing values")
            # Fill numeric with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Remove extreme outliers (houses > $2M or < $50k)
        before_count = len(df_clean)
        df_clean = df_clean[
            (df_clean['SalePrice'] >= 50000) & 
            (df_clean['SalePrice'] <= 2000000)
        ]
        removed = before_count - len(df_clean)
        if removed > 0:
            print(f"   Removed {removed} outliers")
        
        print(f"✅ Data cleaned: {len(df_clean):,} houses remain")
        
        return df_clean
    
    def encode_categoricals(self, df, fit=True):
        """
        Encode categorical variables to numbers
        
        Mental Model:
        Converting words to numbers so computer can understand:
        "Downtown" → 0
        "Suburbs" → 1
        "Rural" → 2
        
        Like translating languages!
        """
        print("📝 Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        categorical_cols = ['Neighborhood', 'GarageType', 'Basement']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df_encoded[col] = self.label_encoders[col].transform(
                            df_encoded[col].astype(str)
                        )
        
        print(f"✅ Encoded {len(categorical_cols)} categorical features")
        
        return df_encoded
    
    def prepare_features(self, df):
        """
        Prepare feature matrix X and target y
        
        Mental Model:
        Separating ingredients:
        X = Everything we measure about the house (features)
        y = What we want to predict (price)
        
        Like separating eggs:
        X = egg whites (inputs)
        y = egg yolks (output we want)
        """
        print("🔧 Preparing features and target...")
        
        # Separate features from target
        feature_cols = [col for col in df.columns if col != 'SalePrice']
        
        X = df[feature_cols].values
        y = df['SalePrice'].values
        
        feature_names = feature_cols
        
        print(f"✅ Features: {X.shape}")
        print(f"   Feature names: {feature_names}")
        print(f"✅ Target: {y.shape}")
        print(f"   Target range: ${y.min():,.0f} - ${y.max():,.0f}")
        
        return X, y, feature_names
    
    def scale_features(self, X_train, X_test, fit=True):
        """
        Scale features to same range
        
        Mental Model:
        Converting different units to same scale:
        
        Before scaling:
          SquareFeet: 500-5000 (huge range)
          Bedrooms: 1-5 (small range)
          Age: 0-74 (medium range)
        
        After scaling:
          SquareFeet: -1.5 to 2.3
          Bedrooms: -1.2 to 1.8
          Age: -1.0 to 2.5
          
        All features now comparable!
        
        Why? Models care about "how much does this feature matter?"
        Without scaling, large numbers dominate just because they're big.
        """
        print("📏 Scaling features...")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"✅ Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load saved preprocessor"""
        preprocessor_data = joblib.load(filepath)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        print(f"✅ Preprocessor loaded from {filepath}")


def main():
    """
    Main preprocessing pipeline
    
    Mental Model:
    Assembly line for data preparation:
    
    Raw data → Load → Clean → Encode → Split → Scale → Save
       ↓        ↓      ↓       ↓        ↓       ↓       ↓
    CSV file   DF    Fixed   Numbers  Train/  Same    Ready!
                                      Test    scale
    """
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - DATA PREPROCESSING")
    print("="*70 + "\n")
    
    # Initialize preprocessor
    preprocessor = HousePricePreprocessor()
    
    # Load data
    df = preprocessor.load_data()
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Encode categoricals
    df = preprocessor.encode_categoricals(df, fit=True)
    
    # Prepare features
    X, y, feature_names = preprocessor.prepare_features(df)
    
    # Split data (80/20)
    print("\n🔪 Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Training set: {len(X_train):,} houses")
    print(f"   Price range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    print(f"   Average: ${y_train.mean():,.0f}")
    
    print(f"✅ Test set: {len(X_test):,} houses")
    print(f"   Price range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    print(f"   Average: ${y_test.mean():,.0f}")
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(
        X_train, X_test, fit=True
    )
    
    # Save processed data
    print("\n💾 Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print("✅ Processed data saved to data/processed/")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n✅ PREPROCESSING COMPLETE!\n")
    
    # Summary
    print("="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Total houses: {len(df):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Average price: ${y.mean():,.0f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
