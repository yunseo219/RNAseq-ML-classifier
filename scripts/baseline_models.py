import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """Build and evaluate baseline models for MCI conversion prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def prepare_mci_data(self):
        """Load and prepare MCI conversion data"""
        print("📥 Loading processed data...")
        
        # Load processed data
        expr = pd.read_csv('data/processed/expression_processed.csv.gz', 
                          compression='gzip', index_col=0)
        meta = pd.read_csv('data/processed/metadata_processed.csv', index_col=0)
        
        # Focus on MCI samples only
        mci_mask = meta['diagnosis'].isin(['MCI', 'MCI_converter'])
        X = expr[mci_mask]
        y = (meta[mci_mask]['diagnosis'] == 'MCI_converter').astype(int)
        
        print(f"✅ MCI Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Class distribution: {y.value_counts().to_dict()}")
        print(f"   Conversion rate: {y.mean()*100:.1f}%")
        
        return X, y, meta[mci_mask]
    
    def handle_class_imbalance(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance"""
        print("\n⚖️ Handling class imbalance with SMOTE...")
        
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"   Original: {len(y_train)} samples")
        print(f"   After SMOTE: {len(y_resampled)} samples")
        print(f"   New class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_baseline_models(self, X, y, use_smote=True):
        """Train multiple baseline models"""
        print("\n"+"="*60)
        print("🤖 TRAINING BASELINE MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nData split: {len(X_train)} train, {len(X_test)} test")
        
        # Handle imbalance if requested
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(probability=True, random_state=self.random_state)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train
            model.fit(X_train_balanced, y_train_balanced)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'auc': auc_score,
                'X_test': X_test
            }
            
            print(f"   AUC: {auc_score:.3f}")
            
            # Print classification report
            print("\n   Classification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Non-converter', 'Converter']))
    
    def cross_validation_evaluation(self, X, y):
        """Perform cross-validation for robust evaluation"""
        print("\n"+"="*60)
        print("🔄 CROSS-VALIDATION EVALUATION")
        print("="*60)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(probability=True, random_state=self.random_state)
        }
        
        cv_results = {}
        
        for name, model in models.items():
            print(f"\n{name}:")
            
            # Cross-validation scores
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            cv_results[name] = scores
            
            print(f"  AUC scores: {scores}")
            print(f"  Mean AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.cv_results = cv_results
        
        return cv_results
    
    def plot_results(self):
        """Create visualization of model performance"""
        print("\n📊 Creating visualizations...")
        
        Path('results').mkdir(exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: ROC curves
        ax = axes[0, 0]
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={results['auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - MCI Conversion Prediction')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: AUC comparison bar plot
        ax = axes[0, 1]
        model_names = list(self.results.keys())
        aucs = [self.results[name]['auc'] for name in model_names]
        bars = ax.bar(model_names, aucs, color=['blue', 'green', 'orange'])
        ax.set_ylabel('AUC Score')
        ax.set_title('Model Comparison - AUC Scores')
        ax.set_ylim([0, 1])
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{auc:.3f}', ha='center')
        
        # Plot 3: Cross-validation results
        ax = axes[0, 2]
        if hasattr(self, 'cv_results'):
            cv_data = []
            cv_labels = []
            for name, scores in self.cv_results.items():
                cv_data.extend(scores)
                cv_labels.extend([name] * len(scores))
            
            import pandas as pd
            cv_df = pd.DataFrame({'Model': cv_labels, 'AUC': cv_data})
            sns.boxplot(data=cv_df, x='Model', y='AUC', ax=ax)
            ax.set_title('Cross-Validation Results')
            ax.set_ylim([0, 1])
        
        # Plot 4-6: Confusion matrices
        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[1, idx]
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-converter', 'Converter'],
                       yticklabels=['Non-converter', 'Converter'])
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('results/baseline_models_performance.png', dpi=150)
        plt.show()
        
        print("✅ Plots saved to results/baseline_models_performance.png")
    
    def save_predictions(self):
        """Save model predictions for later analysis"""
        print("\n💾 Saving predictions...")
        
        Path('results/predictions').mkdir(parents=True, exist_ok=True)
        
        for name, results in self.results.items():
            pred_df = pd.DataFrame({
                'sample_id': results['X_test'].index,
                'true_label': results['y_test'].values,
                'predicted_label': results['y_pred'],
                'predicted_probability': results['y_pred_proba']
            })
            
            filename = f"results/predictions/{name.replace(' ', '_')}_predictions.csv"
            pred_df.to_csv(filename, index=False)
            print(f"   Saved {name} predictions to {filename}")

# Main execution
if __name__ == "__main__":
    # Initialize
    baseline = BaselineModels(random_state=42)
    
    # Load data
    X, y, meta = baseline.prepare_mci_data()
    
    # Train models with SMOTE
    baseline.train_baseline_models(X, y, use_smote=True)
    
    # Cross-validation
    baseline.cross_validation_evaluation(X, y)
    
    # Visualize results
    baseline.plot_results()
    
    # Save predictions
    baseline.save_predictions()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 BASELINE RESULTS SUMMARY")
    print("="*60)
    
    for name, results in baseline.results.items():
        print(f"\n{name}:")
        print(f"  Test AUC: {results['auc']:.3f}")
    
    print("\n✅ Baseline models complete!")