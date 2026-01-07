import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RNASeqPreprocessor:
    """Complete preprocessing pipeline for RNA-seq data"""
    
    def __init__(self, min_expression=1.0, min_samples_expressed=0.1):
        self.min_expression = min_expression
        self.min_samples_expressed = min_samples_expressed
        self.scaler = None
        self.selected_genes = None
        
    def filter_low_expression_genes(self, expr_df):
        """Remove genes with low expression across samples"""
        print("🧬 Filtering low expression genes...")
        
        # Count how many samples express each gene above threshold
        expressed = (expr_df > self.min_expression).sum(axis=0)
        min_samples = len(expr_df) * self.min_samples_expressed
        
        # Keep genes expressed in at least min_samples
        keep_genes = expressed >= min_samples
        
        print(f"  Removed {(~keep_genes).sum()} low-expression genes")
        print(f"  Kept {keep_genes.sum()} genes")
        
        return expr_df.loc[:, keep_genes]
    
    def remove_zero_variance_genes(self, expr_df):
        """Remove genes with zero or near-zero variance"""
        print("📊 Removing zero-variance genes...")
        
        variances = expr_df.var()
        keep_genes = variances > 1e-6
        
        print(f"  Removed {(~keep_genes).sum()} zero-variance genes")
        print(f"  Kept {keep_genes.sum()} genes")
        
        return expr_df.loc[:, keep_genes]
    
    def log_transform(self, expr_df):
        """Log2 transform expression values"""
        print("📈 Log2 transforming expression values...")
        
        # Ensure all values are positive before log transform
        # Replace any negative values with small positive value
        expr_positive = expr_df.copy()
        expr_positive[expr_positive <= 0] = 0.01
        
        # Log2 transform
        expr_log = np.log2(expr_positive + 1)
        
        print(f"  Original range: [{expr_df.min().min():.2f}, {expr_df.max().max():.2f}]")
        print(f"  Log range: [{expr_log.min().min():.2f}, {expr_log.max().max():.2f}]")
        
        # Check for NaN values
        if expr_log.isnull().any().any():
            print("  ⚠️ Found NaN values after log transform, replacing with 0")
            expr_log = expr_log.fillna(0)
        
        return expr_log
    
    def normalize_samples(self, expr_df, method='robust'):
        """Normalize expression values across samples"""
        print(f"🔧 Normalizing with {method} scaling...")
        
        # Handle any remaining NaN values
        if expr_df.isnull().any().any():
            print("  ⚠️ Found NaN values, replacing with median")
            expr_df = expr_df.fillna(expr_df.median())
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Scale genes (features)
        expr_normalized = pd.DataFrame(
            self.scaler.fit_transform(expr_df.T).T,
            index=expr_df.index,
            columns=expr_df.columns
        )
        
        return expr_normalized
    
    def select_variable_genes(self, expr_df, n_genes=5000):
        """Select most variable genes"""
        print(f"🎯 Selecting top {n_genes} variable genes...")
        
        # Handle any infinite values
        expr_df = expr_df.replace([np.inf, -np.inf], np.nan)
        expr_df = expr_df.fillna(expr_df.median())
        
        # Calculate coefficient of variation for each gene
        gene_means = expr_df.mean()
        gene_stds = expr_df.std()
        
        # Avoid division by zero
        gene_cv = gene_stds / (gene_means.abs() + 1e-8)
        
        # Remove genes with infinite CV
        gene_cv = gene_cv[np.isfinite(gene_cv)]
        
        # Select top variable genes
        top_genes = gene_cv.nlargest(n_genes).index
        self.selected_genes = top_genes
        
        print(f"  Selected {len(top_genes)} most variable genes")
        print(f"  CV range: [{gene_cv[top_genes].min():.2f}, {gene_cv[top_genes].max():.2f}]")
        
        return expr_df[top_genes]
    
    def detect_outliers(self, expr_df, meta_df):
        """Detect outlier samples using PCA"""
        print("🔍 Detecting outlier samples...")
        
        # Final check for NaN/Inf values
        expr_clean = expr_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        if expr_clean.isnull().any().any():
            print("  Handling missing values before PCA...")
            expr_clean = expr_clean.fillna(expr_clean.median())
        
        # PCA for outlier detection
        pca = PCA(n_components=2)
        try:
            pca_coords = pca.fit_transform(expr_clean)
        except Exception as e:
            print(f"  ⚠️ PCA failed: {e}")
            print("  Skipping outlier detection")
            return np.ones(len(expr_df), dtype=bool)  # Keep all samples
        
        # Calculate distance from center
        center = np.median(pca_coords, axis=0)
        distances = np.sqrt(np.sum((pca_coords - center)**2, axis=1))
        
        # Flag outliers (>3 std from median)
        outlier_threshold = np.median(distances) + 3 * np.std(distances)
        outliers = distances > outlier_threshold
        
        print(f"  Found {outliers.sum()} outlier samples")
        if outliers.sum() > 0:
            outlier_samples = expr_df.index[outliers]
            outlier_diagnoses = meta_df.loc[outlier_samples, 'diagnosis'].value_counts()
            print(f"  Outliers by diagnosis: {outlier_diagnoses.to_dict()}")
        
        # Create visualization
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                               c=['red' if o else 'blue' for o in outliers],
                               alpha=0.6)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title('Sample Outlier Detection via PCA')
            
            # Create results directory if it doesn't exist
            Path('results').mkdir(exist_ok=True)
            plt.savefig('results/outlier_detection.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  Saved outlier plot to results/outlier_detection.png")
        except Exception as e:
            print(f"  Could not create visualization: {e}")
        
        return ~outliers  # Return mask of samples to keep
    
    def full_preprocessing_pipeline(self, expr_df, meta_df, 
                                   filter_outliers=True, n_variable_genes=5000):
        """Run complete preprocessing pipeline"""
        print("="*60)
        print("🚀 RUNNING FULL PREPROCESSING PIPELINE")
        print("="*60)
        
        print(f"\n📥 Input: {expr_df.shape[0]} samples, {expr_df.shape[1]} genes")
        
        # 1. Filter low expression genes
        expr_df = self.filter_low_expression_genes(expr_df)
        
        # 2. Remove zero variance genes
        expr_df = self.remove_zero_variance_genes(expr_df)
        
        # 3. Log transform (with fixes for negative values)
        expr_df = self.log_transform(expr_df)
        
        # 4. Normalize
        expr_df = self.normalize_samples(expr_df, method='robust')
        
        # 5. Select variable genes
        expr_df = self.select_variable_genes(expr_df, n_genes=n_variable_genes)
        
        # 6. Detect and optionally remove outliers
        if filter_outliers:
            keep_samples = self.detect_outliers(expr_df, meta_df)
            expr_df = expr_df[keep_samples]
            meta_df = meta_df[keep_samples]
            print(f"\n📤 After outlier removal: {expr_df.shape[0]} samples")
        
        print(f"\n✅ Final: {expr_df.shape[0]} samples, {expr_df.shape[1]} genes")
        
        return expr_df, meta_df

# Run preprocessing
if __name__ == "__main__":
    from scripts.download_manager_v2 import RobustDataManager
    
    print("Loading data...")
    
    # Load data
    manager = RobustDataManager(use_aws=True, bucket_name='ad-rnaseq-prediction-data')
    expr, meta = manager.load_from_s3('GSE63061')
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing
    preprocessor = RNASeqPreprocessor()
    expr_processed, meta_processed = preprocessor.full_preprocessing_pipeline(
        expr, meta, 
        filter_outliers=True,
        n_variable_genes=5000
    )
    
    # Save processed data
    print("\n💾 Saving processed data...")
    expr_processed.to_csv('data/processed/expression_processed.csv.gz', compression='gzip')
    meta_processed.to_csv('data/processed/metadata_processed.csv')
    
    print("✅ Processed data saved to data/processed/")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"  Final samples: {expr_processed.shape[0]}")
    print(f"  Final genes: {expr_processed.shape[1]}")
    print(f"  Diagnosis distribution:")
    print(meta_processed['diagnosis'].value_counts())