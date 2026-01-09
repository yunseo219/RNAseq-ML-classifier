import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedModels:
    """Advanced ML models with hyperparameter tuning"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.results = {}
    
    def load_data(self):
        """Load processed MCI data"""
        print("📥 Loading data...")
        
        expr = pd.read_csv('data/processed/expression_processed.csv.gz', 
                          compression='gzip', index_col=0)
        meta = pd.read_csv('data/processed/metadata_processed.csv', index_col=0)
        
        # MCI samples only
        mci_mask = meta['diagnosis'].isin(['MCI', 'MCI_converter'])
        X = expr[mci_mask]
        y = (meta[mci_mask]['diagnosis'] == 'MCI_converter').astype(int)
        
        print(f"✅ Data shape: {X.shape}")
        print(f"   Conversion rate: {y.mean()*100:.1f}%")
        
        return X, y
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with hyperparameter tuning"""
        print("\n🚀 Training XGBoost...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'scale_pos_weight': [1, 3]  # Handle imbalance
        }
        
        xgb = XGBClassifier(random_state=self.random_state, use_label_encoder=False)
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            xgb, param_grid, cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_xgb = grid_search.best_estimator_
        
        # Evaluate
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   Test AUC: {auc:.3f}")
        
        self.best_models['XGBoost'] = best_xgb
        self.results['XGBoost'] = {
            'model': best_xgb,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid_search.best_params_
        }
        
        return best_xgb
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting"""
        print("\n🌲 Training Gradient Boosting...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        gb = GradientBoostingClassifier(random_state=self.random_state)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            gb, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_gb = grid_search.best_estimator_
        
        y_pred = best_gb.predict(X_test)
        y_pred_proba = best_gb.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   Test AUC: {auc:.3f}")
        
        self.best_models['GradientBoosting'] = best_gb
        self.results['GradientBoosting'] = {
            'model': best_gb,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid_search.best_params_
        }
        
        return best_gb
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network"""
        print("\n🧠 Training Neural Network...")
        
        # Scale features for NN
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.001, 0.01],
            'learning_rate': ['adaptive']
        }
        
        nn = MLPClassifier(max_iter=1000, random_state=self.random_state)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            nn, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        best_nn = grid_search.best_estimator_
        
        y_pred = best_nn.predict(X_test_scaled)
        y_pred_proba = best_nn.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   Test AUC: {auc:.3f}")
        
        self.best_models['NeuralNetwork'] = best_nn
        self.results['NeuralNetwork'] = {
            'model': best_nn,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': grid_search.best_params_,
            'scaler': scaler
        }
        
        return best_nn
    
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Create ensemble model"""
        print("\n🎭 Creating Ensemble Model...")
        
        # Use best models from previous training
        if len(self.best_models) < 2:
            print("   Need at least 2 models for ensemble")
            return None
        
        # Create voting classifier
        estimators = [(name, model) for name, model in self.best_models.items()
                     if name != 'NeuralNetwork']  # Exclude NN due to scaling
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   Ensemble AUC: {auc:.3f}")
        
        self.best_models['Ensemble'] = ensemble
        self.results['Ensemble'] = {
            'model': ensemble,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return ensemble
    
    def compare_models(self):
        """Create comparison visualization"""
        print("\n📊 Creating model comparison...")
        
        Path('results').mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # AUC comparison
        ax = axes[0]
        models = list(self.results.keys())
        aucs = [self.results[m]['auc'] for m in models]
        
        bars = ax.bar(models, aucs, color=['blue', 'green', 'orange', 'purple', 'red'][:len(models)])
        ax.set_ylabel('AUC Score')
        ax.set_title('Advanced Models - AUC Comparison')
        ax.set_ylim([0, 1])
        
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{auc:.3f}', ha='center')
        
        # Feature importance (for XGBoost)
        ax = axes[1]
        if 'XGBoost' in self.best_models:
            xgb_model = self.best_models['XGBoost']
            importances = xgb_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20
            
            ax.bar(range(20), importances[indices])
            ax.set_title('Top 20 Feature Importances (XGBoost)')
            ax.set_xlabel('Feature Rank')
            ax.set_ylabel('Importance')
        
        plt.tight_layout()
        plt.savefig('results/advanced_models_comparison.png', dpi=150)
        plt.show()
        
        print("✅ Plots saved to results/advanced_models_comparison.png")
    
    def save_best_model(self):
        """Save the best performing model"""
        print("\n💾 Saving best model...")
        
        Path('models').mkdir(exist_ok=True)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['auc'])
        best_model = self.best_models[best_model_name]
        best_auc = self.results[best_model_name]['auc']
        
        # Save model
        joblib.dump(best_model, f'models/best_model_{best_model_name}.pkl')
        
        # Save model info
        model_info = {
            'model_type': best_model_name,
            'auc': best_auc,
            'parameters': self.results[best_model_name].get('best_params', {})
        }
        
        pd.DataFrame([model_info]).to_csv('models/best_model_info.csv', index=False)
        
        print(f"✅ Saved best model: {best_model_name} (AUC: {best_auc:.3f})")

# Main execution
if __name__ == "__main__":
    # Initialize
    advanced = AdvancedModels(random_state=42)
    
    # Load data
    X, y = advanced.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\n📊 Training set after SMOTE: {X_train_balanced.shape}")
    
    # Train advanced models
    advanced.train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)
    advanced.train_gradient_boosting(X_train_balanced, y_train_balanced, X_test, y_test)
    advanced.train_neural_network(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Create ensemble
    advanced.create_ensemble(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Compare models
    advanced.compare_models()
    
    # Save best model
    advanced.save_best_model()
    
    # Print final summary
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)
    
    for name, results in advanced.results.items():
        print(f"{name}: AUC = {results['auc']:.3f}")
    
    best_model = max(advanced.results.items(), key=lambda x: x[1]['auc'])
    print(f"\n🥇 Best Model: {best_model[0]} with AUC = {best_model[1]['auc']:.3f}")