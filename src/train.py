"""
Train regression models for house price prediction

Mental Model:
Training models = Teaching different students the same material
- Some learn patterns quickly (Linear Regression)
- Some need to see complex relationships (Random Forest)
- Some learn from mistakes iteratively (Gradient Boosting)

We train all of them, see who performs best!
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """
    Load preprocessed data
    
    Mental Model:
    Opening our prepared ingredients from the fridge
    Everything is already cleaned, chopped, measured
    Ready to cook!
    """
    print("📥 Loading processed data...")
    
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Data loaded successfully")
    print(f"   Training set: {len(X_train):,} houses")
    print(f"   Test set: {len(X_test):,} houses")
    print(f"   Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names

def get_regression_models():
    """
    Initialize regression models to compare
    
    Mental Model:
    Assembling a team of price estimators:
    Each has different approach to estimating house prices
    
    LINEAR REGRESSION:
      "I draw a straight line through the data"
      Pros: Fast, interpretable
      Cons: Can't capture complex patterns
      
    RIDGE REGRESSION:
      "Like Linear but I don't overfit"
      Pros: Handles many features well
      Cons: Still linear
      
    LASSO REGRESSION:
      "I also remove useless features automatically"
      Pros: Feature selection built-in
      Cons: Can be too aggressive
      
    DECISION TREE:
      "I ask yes/no questions about features"
      Pros: Handles non-linear patterns
      Cons: Can overfit easily
      
    RANDOM FOREST:
      "I'm 100 decision trees voting together"
      Pros: Very accurate, robust
      Cons: Slower, less interpretable
      
    GRADIENT BOOSTING:
      "I learn from my mistakes sequentially"
      Pros: Often best performance
      Cons: Slower training
      
    XGBOOST:
      "I'm Gradient Boosting on steroids"
      Pros: Wins competitions
      Cons: Many parameters to tune
    """
    print("\n🤖 Initializing regression models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        
        'Ridge Regression': Ridge(
            alpha=10.0,
            random_state=42
        ),
        
        'Lasso Regression': Lasso(
            alpha=10.0,
            random_state=42,
            max_iter=10000
        ),
        
        'Decision Tree': DecisionTreeRegressor(
            random_state=42,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        ),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.1
        ),
        
        'XGBoost': XGBRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
    }
    
    print(f"✅ Initialized {len(models)} models")
    
    return models

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Train all models and evaluate with REGRESSION metrics
    
    Mental Model:
    Like a cooking competition:
    
    Each chef (model) gets same ingredients (training data)
    They prepare their dish (train)
    Judges taste and score (evaluate on test data)
    
    Scoring criteria (regression metrics):
    - R² Score: "How well does it explain variation?"
    - MAE: "On average, how many dollars off?"
    - RMSE: "Any really bad predictions?"
    """
    results = []
    trained_models = {}
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION (REGRESSION)")
    print("="*70 + "\n")
    
    for name, model in models.items():
        print(f"🤖 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate REGRESSION metrics
        # R² Score (0 to 1, higher is better)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # MAE (Mean Absolute Error - in dollars)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # RMSE (Root Mean Squared Error - in dollars)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results.append({
            'Model': name,
            'R²_Train': r2_train,
            'R²_Test': r2_test,
            'MAE_Train': mae_train,
            'MAE_Test': mae_test,
            'RMSE_Train': rmse_train,
            'RMSE_Test': rmse_test,
            'Overfit_Gap': r2_train - r2_test
        })
        
        trained_models[name] = {
            'model': model,
            'predictions': y_pred_test
        }
        
        print(f"   ✅ R² (Train): {r2_train:.4f} | R² (Test): {r2_test:.4f}")
        print(f"      MAE (Test): ${mae_test:,.0f} | RMSE (Test): ${rmse_test:,.0f}")
        print(f"      Overfitting gap: {r2_train - r2_test:.4f}\n")
    
    results_df = pd.DataFrame(results)
    
    return results_df, trained_models

def plot_model_comparison(results_df):
    """
    Visualize model comparison
    
    Mental Model:
    Like showing competition scoreboard:
    See which model performed best at a glance
    """
    print("📊 Creating model comparison plots...")
    
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: R² Score Comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x_pos - width/2, results_df['R²_Train'], width, 
           label='Train', color='lightblue', edgecolor='black', alpha=0.8)
    ax.bar(x_pos + width/2, results_df['R²_Test'], width,
           label='Test', color='coral', edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax.set_ylabel('R² Score', fontweight='bold', fontsize=11)
    ax.set_title('R² Score Comparison (Higher is Better)', 
                 fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Plot 2: MAE Comparison
    ax = axes[0, 1]
    ax.barh(results_df['Model'], results_df['MAE_Test'], 
            color='mediumseagreen', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Mean Absolute Error ($)', fontweight='bold', fontsize=11)
    ax.set_title('MAE Comparison (Lower is Better)', 
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(results_df['MAE_Test']):
        ax.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)
    
    # Plot 3: RMSE Comparison
    ax = axes[1, 0]
    ax.barh(results_df['Model'], results_df['RMSE_Test'],
            color='orchid', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Root Mean Squared Error ($)', fontweight='bold', fontsize=11)
    ax.set_title('RMSE Comparison (Lower is Better)',
                 fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(results_df['RMSE_Test']):
        ax.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)
    
    # Plot 4: Overfitting Analysis
    ax = axes[1, 1]
    colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' 
              for gap in results_df['Overfit_Gap']]
    ax.barh(results_df['Model'], results_df['Overfit_Gap'],
            color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Train R² - Test R² (Gap)', fontweight='bold', fontsize=11)
    ax.set_title('Overfitting Analysis (Lower is Better)',
                 fontweight='bold', fontsize=12)
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, 
               label='Warning (0.1)')
    ax.axvline(x=0.2, color='red', linestyle='--', linewidth=2,
               label='Severe (0.2)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Regression Model Performance Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/model_comparison.png")
    plt.close()

def plot_predictions_vs_actual(trained_models, y_test):
    """
    Plot predicted vs actual prices for all models
    
    Mental Model:
    Scatter plot showing:
    - X-axis: What the house ACTUALLY sold for
    - Y-axis: What the model PREDICTED
    - Perfect predictions = all points on diagonal line
    - Points above line = overestimated
    - Points below line = underestimated
    """
    print("📊 Creating actual vs predicted plots...")
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, (name, data) in enumerate(trained_models.items()):
        y_pred = data['predictions']
        
        # Scatter plot
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20, color='blue')
        
        # Perfect prediction line (diagonal)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                      'r--', lw=2, label='Perfect Predictions')
        
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        axes[idx].set_xlabel('Actual Price ($)', fontweight='bold')
        axes[idx].set_ylabel('Predicted Price ($)', fontweight='bold')
        axes[idx].set_title(f'{name}\nR²={r2:.4f}, MAE=${mae:,.0f}',
                           fontweight='bold')
        axes[idx].legend(loc='upper left')
        axes[idx].grid(alpha=0.3)
    
    # Hide last subplot
    axes[-1].axis('off')
    
    plt.suptitle('Actual vs Predicted Prices - All Models',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/actual_vs_predicted.png")
    plt.close()

def plot_residuals(trained_models, y_test):
    """
    Plot residual plots (errors) for all models
    
    Mental Model:
    Residual = Actual - Predicted (the error)
    
    Good residual plot:
    - Points randomly scattered around 0
    - No patterns (no curve, no fan shape)
    - Means errors are random, not systematic
    
    Bad residual plot:
    - Curve shape = model missing non-linear pattern
    - Fan shape = variance increases with price
    - Clustering = different house types need different models
    """
    print("📊 Creating residual plots...")
    
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, (name, data) in enumerate(trained_models.items()):
        y_pred = data['predictions']
        residuals = y_test - y_pred
        
        # Scatter plot of residuals
        axes[idx].scatter(y_pred, residuals, alpha=0.5, s=20, color='green')
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        
        axes[idx].set_xlabel('Predicted Price ($)', fontweight='bold')
        axes[idx].set_ylabel('Residual (Actual - Predicted) ($)', fontweight='bold')
        axes[idx].set_title(f'{name}\nResidual Plot', fontweight='bold')
        axes[idx].grid(alpha=0.3)
        
        # Add horizontal lines at ±1 std
        std_res = residuals.std()
        axes[idx].axhline(y=std_res, color='orange', linestyle=':', lw=1.5,
                         label=f'±1 std (${std_res:,.0f})')
        axes[idx].axhline(y=-std_res, color='orange', linestyle=':', lw=1.5)
        axes[idx].legend()
    
    # Hide last subplot
    axes[-1].axis('off')
    
    plt.suptitle('Residual Plots - All Models (Good: Random scatter around 0)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/residual_plots.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/residual_plots.png")
    plt.close()

def save_best_model(results_df, trained_models, feature_names):
    """
    Save the best performing model
    
    Mental Model:
    Like picking the MVP (Most Valuable Player):
    - Look at R² score (how well it explains variation)
    - Check overfitting (train vs test gap)
    - Consider MAE (practical error amount)
    
    Usually pick based on highest Test R² with low overfitting
    """
    print("\n💾 Selecting and saving best model...")
    
    # Find best model by Test R² score
    best_idx = results_df['R²_Test'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]['model']
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'metrics': results_df.loc[best_idx].to_dict(),
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'problem_type': 'regression'
    }
    joblib.dump(metadata, 'models/model_metadata.pkl')
    
    print(f"✅ Best model: {best_model_name}")
    print(f"   R² (Test): {results_df.loc[best_idx, 'R²_Test']:.4f}")
    print(f"   MAE (Test): ${results_df.loc[best_idx, 'MAE_Test']:,.0f}")
    print(f"   RMSE (Test): ${results_df.loc[best_idx, 'RMSE_Test']:,.0f}")
    print(f"   Overfitting gap: {results_df.loc[best_idx, 'Overfit_Gap']:.4f}")
    print(f"   Saved to models/best_model.pkl")
    
    return best_model_name, best_model

def main():
    """
    Main training pipeline
    
    Mental Model:
    Complete cooking competition:
    1. Get ingredients (load data)
    2. Assemble chefs (initialize models)
    3. Cook (train)
    4. Taste test (evaluate)
    5. Show results (visualize)
    6. Crown winner (save best model)
    """
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING (REGRESSION)")
    print("="*70 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    # Get models
    models = get_regression_models()
    
    # Train and evaluate
    results_df, trained_models = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )
    
    # Display results
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS (REGRESSION METRICS)")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_results.csv', index=False)
    print("✅ Results saved to results/model_results.csv")
    
    # Create visualizations
    plot_model_comparison(results_df)
    plot_predictions_vs_actual(trained_models, y_test)
    plot_residuals(trained_models, y_test)
    
    # Save best model
    best_model_name, best_model = save_best_model(
        results_df, trained_models, feature_names
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📁 Saved to: models/best_model.pkl")
    print(f"📊 Results saved to: results/")
    print(f"📈 Visualizations created in: results/\n")
    
    # Mental Model Summary
    print("="*70)
    print("MENTAL MODEL: What Just Happened?")
    print("="*70)
    print("""
We trained 7 different "price estimators" (models):
  1. Linear Regression - Drew straight line through data
  2. Ridge - Linear but prevents overfitting
  3. Lasso - Linear + automatic feature selection
  4. Decision Tree - Asked yes/no questions
  5. Random Forest - 100 decision trees voting
  6. Gradient Boosting - Learned from mistakes
  7. XGBoost - Advanced gradient boosting

Each model learned patterns like:
  "Bigger house → higher price"
  "Better quality → higher price"
  "Older house → lower price"
  "Better neighborhood → higher price"

We evaluated using REGRESSION metrics:
  R² Score: "How much variation can I explain?"
  MAE: "On average, how many dollars off?"
  RMSE: "Any really bad predictions?"

Winner: The model with highest R² and low error!
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()