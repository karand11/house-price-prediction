"""
Detailed evaluation of best regression model

Mental Model:
Like a detailed restaurant review:
- Overall rating (R²)
- Specific dishes (feature importance)
- What went wrong (error analysis)
- Customer feedback (residual patterns)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

def load_model_and_data():
    """Load best model and test data"""
    print("📥 Loading model and data...")
    
    # Load model
    model = joblib.load('models/best_model.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    
    # Load test data
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Load feature names
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Loaded model: {metadata['model_name']}")
    print(f"   Test samples: {len(X_test):,}")
    
    return model, X_test, y_test, feature_names, metadata

def detailed_metrics_report(model, X_test, y_test):
    """
    Generate detailed regression metrics
    
    Mental Model:
    Like a comprehensive health checkup:
    - Overall health (R²)
    - Average error (MAE)
    - Worst-case scenarios (RMSE, Max Error)
    - Distribution of errors (percentiles)
    """
    print("\n" + "="*70)
    print("DETAILED REGRESSION METRICS")
    print("="*70 + "\n")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    
    # Additional metrics
    errors = np.abs(y_test - y_pred)
    max_error = errors.max()
    min_error = errors.min()
    median_error = np.median(errors)
    
    # Percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Error percentiles
    p25_error = np.percentile(errors, 25)
    p50_error = np.percentile(errors, 50)
    p75_error = np.percentile(errors, 75)
    p95_error = np.percentile(errors, 95)
    
    print("📊 REGRESSION PERFORMANCE METRICS:")
    print("="*70)
    print(f"\n1. R² Score (Coefficient of Determination):")
    print(f"   Value: {r2:.4f}")
    print(f"   Interpretation: Model explains {r2*100:.2f}% of price variation")
    if r2 > 0.9:
        print(f"   Rating: ⭐⭐⭐⭐⭐ Excellent!")
    elif r2 > 0.8:
        print(f"   Rating: ⭐⭐⭐⭐ Very Good")
    elif r2 > 0.7:
        print(f"   Rating: ⭐⭐⭐ Good")
    else:
        print(f"   Rating: ⭐⭐ Needs Improvement")
    
    print(f"\n2. Mean Absolute Error (MAE):")
    print(f"   Value: ${mae:,.0f}")
    print(f"   Interpretation: On average, predictions are ${mae:,.0f} off")
    print(f"   As % of avg price: {mae/y_test.mean()*100:.2f}%")
    
    print(f"\n3. Root Mean Squared Error (RMSE):")
    print(f"   Value: ${rmse:,.0f}")
    print(f"   Interpretation: Penalizes large errors more than MAE")
    print(f"   RMSE/MAE ratio: {rmse/mae:.2f} (closer to 1 = consistent errors)")
    
    print(f"\n4. Mean Squared Error (MSE):")
    print(f"   Value: ${mse:,.0f}")
    
    print(f"\n5. Mean Absolute Percentage Error (MAPE):")
    print(f"   Value: {mape:.2f}%")
    print(f"   Interpretation: Average error is {mape:.2f}% of actual price")
    
    print(f"\n6. Error Range:")
    print(f"   Minimum error: ${min_error:,.0f}")
    print(f"   Maximum error: ${max_error:,.0f}")
    print(f"   Median error: ${median_error:,.0f}")
    
    print(f"\n7. Error Distribution (Percentiles):")
    print(f"   25th percentile: ${p25_error:,.0f} (25% of errors below this)")
    print(f"   50th percentile: ${p50_error:,.0f} (median)")
    print(f"   75th percentile: ${p75_error:,.0f} (75% of errors below this)")
    print(f"   95th percentile: ${p95_error:,.0f} (95% of errors below this)")
    
    print("\n" + "="*70)
    
    # Save report
    with open('results/detailed_metrics.txt', 'w') as f:
        f.write("DETAILED REGRESSION METRICS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"MAE: ${mae:,.0f}\n")
        f.write(f"RMSE: ${rmse:,.0f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"Max Error: ${max_error:,.0f}\n")
    
    print("✅ Detailed report saved to results/detailed_metrics.txt\n")
    
    return y_pred

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models
    
    Mental Model:
    Which house features matter most for price?
    
    Like asking: "What makes a house valuable?"
    - Location? (Neighborhood)
    - Size? (Square Feet)
    - Quality? (Overall Quality)
    - Age? (Year Built)
    
    The model tells us what it learned!
    """
    print("📊 Analyzing feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_n = min(15, len(feature_names))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(top_n), list(top_features['Importance'].values),
            color='teal', edgecolor='black', alpha=0.8)
        plt.yticks(range(top_n), list(top_features['Feature'].values))
        plt.xlabel('Importance Score', fontweight='bold', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features for House Price',
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✅ Saved to results/feature_importance.png")
        plt.close()
        
        # Save to CSV
        importance_df.to_csv('results/feature_importance.csv', index=False)
        print("✅ Feature importance saved to results/feature_importance.csv")
        
        # Print top features
        print("\n🔝 Top 10 Most Important Features:")
        print("="*70)
        for i in range(min(10, len(importance_df))):
            row = importance_df.iloc[i]
            print(f"{i+1:2d}. {row['Feature']:20s} {row['Importance']:.4f}")
        print("="*70 + "\n")
    else:
        print("⚠️  Model doesn't have feature_importances_ attribute\n")

def analyze_prediction_errors(y_test, y_pred):
    """
    Analyze where predictions go wrong
    
    Mental Model:
    Like a doctor analyzing treatment failures:
    - Which patients (houses) did we mistreat (predict badly)?
    - What do they have in common?
    - Can we learn from our mistakes?
    """
    print("🔍 Analyzing prediction errors...")
    
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / y_test) * 100
    
    # Find worst predictions
    worst_indices = np.argsort(abs_errors)[::-1][:10]
    
    print("\n❌ Top 10 Worst Predictions:")
    print("="*70)
    print(f"{'Rank':<6} {'Actual Price':<15} {'Predicted':<15} {'Error':<15} {'% Error':<10}")
    print("-"*70)
    for i, idx in enumerate(worst_indices, 1):
        print(f"{i:<6} ${y_test[idx]:>13,.0f} ${y_pred[idx]:>13,.0f} "
              f"${abs_errors[idx]:>13,.0f} {pct_errors[idx]:>8.1f}%")
    print("="*70 + "\n")
    
    # Categorize errors
    small_error = (abs_errors < 20000).sum()
    medium_error = ((abs_errors >= 20000) & (abs_errors < 50000)).sum()
    large_error = (abs_errors >= 50000).sum()
    
    total = len(y_test)
    
    print("📊 Error Distribution:")
    print("="*70)
    print(f"Small errors (<$20k):   {small_error:4d} ({small_error/total*100:5.1f}%)")
    print(f"Medium errors ($20-50k): {medium_error:4d} ({medium_error/total*100:5.1f}%)")
    print(f"Large errors (>$50k):    {large_error:4d} ({large_error/total*100:5.1f}%)")
    print("="*70 + "\n")

def plot_error_distribution(y_test, y_pred):
    """
    Plot distribution of prediction errors
    
    Mental Model:
    Histogram of errors shows:
    - Is error centered at 0? (unbiased)
    - Are errors symmetric? (no systematic over/under prediction)
    - Are there outliers? (very bad predictions)
    """
    print("📊 Creating error distribution plot...")
    
    errors = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect (0 error)')
    axes[0].axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean error: ${errors.mean():,.0f}')
    axes[0].set_xlabel('Prediction Error (Actual - Predicted) ($)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Distribution of Prediction Errors', fontweight='bold', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot([errors], labels=['Errors'], vert=True)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Prediction Error ($)', fontweight='bold')
    axes[1].set_title('Error Box Plot (Shows Outliers)', fontweight='bold', fontsize=13)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/error_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/error_distribution.png")
    plt.close()

def plot_price_range_performance(y_test, y_pred):
    """
    Analyze performance across different price ranges
    
    Mental Model:
    Does model work equally well for:
    - Cheap houses (<$200k)?
    - Mid-range houses ($200k-$400k)?
    - Expensive houses (>$400k)?
    
    Or does it struggle with certain price ranges?
    """
    print("📊 Analyzing performance by price range...")
    
    # Create price bins
    bins = [0, 200000, 400000, 600000, np.inf]
    labels = ['<$200k', '$200k-$400k', '$400k-$600k', '>$600k']
    
    price_ranges = pd.cut(y_test, bins=bins, labels=labels)
    
    # Calculate metrics for each range
    range_metrics = []
    for label in labels:
        mask = price_ranges == label
        if mask.sum() > 0:
            range_y_test = y_test[mask]
            range_y_pred = y_pred[mask]
            
            range_metrics.append({
                'Price Range': label,
                'Count': mask.sum(),
                'R²': r2_score(range_y_test, range_y_pred),
                'MAE': mean_absolute_error(range_y_test, range_y_pred),
                'MAPE': np.mean(np.abs((range_y_test - range_y_pred) / range_y_test)) * 100
            })
    
    metrics_df = pd.DataFrame(range_metrics)
    
    print("\n📊 Performance by Price Range:")
    print("="*70)
    print(metrics_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² by range
    axes[0].bar(metrics_df['Price Range'], metrics_df['R²'],
               color='coral', edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('R² Score', fontweight='bold')
    axes[0].set_title('R² Score by Price Range', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE by range
    axes[1].bar(metrics_df['Price Range'], metrics_df['MAE'],
               color='lightblue', edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('MAE ($)', fontweight='bold')
    axes[1].set_title('MAE by Price Range', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # MAPE by range
    axes[2].bar(metrics_df['Price Range'], metrics_df['MAPE'],
               color='lightgreen', edgecolor='black', alpha=0.8)
    axes[2].set_ylabel('MAPE (%)', fontweight='bold')
    axes[2].set_title('MAPE by Price Range', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/performance_by_price_range.png', dpi=300, bbox_inches='tight')
    print("✅ Saved to results/performance_by_price_range.png")
    plt.close()

def main():
    """
    Main evaluation pipeline
    
    Mental Model:
    Like a comprehensive restaurant review:
    1. Overall rating (metrics)
    2. Best dishes (feature importance)
    3. What went wrong (error analysis)
    4. Different customer experiences (price range analysis)
    """
    print("\n" + "="*70)
    print("DETAILED MODEL EVALUATION (REGRESSION)")
    print("="*70 + "\n")
    
    # Load model and data
    model, X_test, y_test, feature_names, metadata = load_model_and_data()
    
    # Display model info
    print(f"🤖 Model: {metadata['model_name']}")
    print(f"📅 Trained: {metadata.get('training_date', 'Unknown')}")
    print(f"📊 Training Metrics:")
    for metric, value in metadata['metrics'].items():
        if metric != 'Model' and not metric.startswith('_'):
            if 'MAE' in metric or 'RMSE' in metric:
                print(f"   {metric}: ${value:,.0f}")
            else:
                print(f"   {metric}: {value:.4f}")
    
    # Detailed metrics
    y_pred = detailed_metrics_report(model, X_test, y_test)
    
    # Feature importance
    plot_feature_importance(model, feature_names)
    
    # Error analysis
    analyze_prediction_errors(y_test, y_pred)
    
    # Error distribution
    plot_error_distribution(y_test, y_pred)
    
    # Price range analysis
    plot_price_range_performance(y_test, y_pred)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 All results saved to results/")
    print("   - detailed_metrics.txt")
    print("   - feature_importance.png")
    print("   - feature_importance.csv")
    print("   - error_distribution.png")
    print("   - performance_by_price_range.png\n")
    
    # Mental Model Summary
    print("="*70)
    print("MENTAL MODEL: Understanding Regression Evaluation")
    print("="*70)
    print("""
REGRESSION METRICS vs CLASSIFICATION METRICS:

REGRESSION (House Price):
  Question: "How far off are my predictions?"
  
  R² Score:
    - "What % of variation can I explain?"
    - Range: 0 to 1 (higher = better)
    - 0.85 = Explains 85% of why prices vary
  
  MAE (Mean Absolute Error):
    - "On average, how many dollars off?"
    - $25,000 MAE = Average error is $25k
    - Easy to interpret!
  
  RMSE (Root Mean Squared Error):
    - "Are there any really bad predictions?"
    - Penalizes large errors more
    - RMSE > MAE means some outliers exist

CLASSIFICATION (Churn):
  Question: "Did I put things in right bucket?"
  
  Accuracy: "Overall correctness"
  Precision: "When I say YES, am I right?"
  Recall: "Did I catch all the YES cases?"
  F1-Score: "Balance of precision and recall"

WHY DIFFERENT?
  Regression = predicting NUMBER (can be "close" or "far")
  Classification = predicting CATEGORY (only "right" or "wrong")
  
  You can be "$10k off" in regression
  You can't be "a little bit off" in classification!
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()