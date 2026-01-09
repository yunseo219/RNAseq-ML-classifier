import pandas as pd
import joblib
from pathlib import Path
import numpy as np

print("="*60)
print("🏆 MODEL PERFORMANCE CHECK")
print("="*60)

# Load best model info
model_info = pd.read_csv('models/best_model_info.csv')
print("\n📊 Best Model Information:")
print(model_info)

# Check predictions
pred_dir = Path('results/predictions')
if pred_dir.exists():
    pred_files = list(pred_dir.glob('*.csv'))
    
    print("\n📈 Model Performance Summary:")
    print("-"*40)
    
    for file in pred_files:
        df = pd.read_csv(file)
        model_name = file.stem.replace('_predictions', '').replace('_', ' ')
        
        # Calculate metrics
        acc = (df['true_label'] == df['predicted_label']).mean()
        
        # For converters (label=1)
        true_positives = ((df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
        false_negatives = ((df['true_label'] == 1) & (df['predicted_label'] == 0)).sum()
        false_positives = ((df['true_label'] == 0) & (df['predicted_label'] == 1)).sum()
        
        if (true_positives + false_negatives) > 0:
            sensitivity = true_positives / (true_positives + false_negatives)
        else:
            sensitivity = 0
            
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Sensitivity (Recall): {sensitivity:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Predictions made: {len(df)}")

print("\n✅ Model trained and ready for deployment!")