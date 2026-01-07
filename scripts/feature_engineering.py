import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    """Feature engineering for RNA-seq data"""
    
    def __init__(self):
        self.top_genes = None
        self.gene_scores = None
    
    def differential_expression(self, expr_df, labels, groups=['MCI', 'MCI_converter']):
        """Find differentially expressed genes between groups"""
        print("🧬 Finding differentially expressed genes...")
        
        # Separate groups
        group1_mask = labels == 0  # Non-converters
        group2_mask = labels == 1  # Converters
        
        results = []
        
        for gene in expr_df.columns:
            group1_expr = expr_df.loc[group1_mask, gene]
            group2_expr = expr_df.loc[group2_mask, gene]
            
            # T-test
            t_stat, p_value = stats.ttest_ind(group1_expr, group2_expr)
            
            # Fold change
            mean1 = group1_expr.mean()
            mean2 = group2_expr.mean()
            fold_change = mean2 - mean1  # Log scale
            
            results.append({
                'gene': gene,
                't_statistic': t_stat,
                'p_value': p_value,
                'fold_change': fold_change,
                'mean_non_converter': mean1,
                'mean_converter': mean2
            })
        
        de_results = pd.DataFrame(results)
        de_results['adjusted_p'] = self._adjust_pvalues(de_results['p_value'])
        de_results = de_results.sort_values('p_value')
        
        # Identify significant genes
        sig_genes = de_results[
            (de_results['adjusted_p'] < 0.05) & 
            (np.abs(de_results['fold_change']) > 0.5)
        ]
        
        print(f"  Found {len(sig_genes)} significant genes (adj.p < 0.05, |FC| > 0.5)")
        
        return de_results
    
    def _adjust_pvalues(self, pvalues):
        """Benjamini-Hochberg FDR correction"""
        from statsmodels.stats.multitest import multipletests
        return multipletests(pvalues, method='fdr_bh')[1]
    
    def univariate_selection(self, X, y, n_features=100):
        """Select features using univariate statistics"""
        print(f"📊 Univariate feature selection (top {n_features})...")
        
        # ANOVA F-test
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y)
        
        # Get scores
        scores = pd.DataFrame({
            'gene': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected_genes = scores.head(n_features)['gene'].values
        print(f"  Top gene score: {scores.iloc[0]['score']:.2f}")
        print(f"  {n_features}th gene score: {scores.iloc[n_features-1]['score']:.2f}")
        
        return selected_genes, scores
    
    def random_forest_importance(self, X, y, n_features=100):
        """Select features using Random Forest importance"""
        print(f"🌲 Random Forest feature selection...")
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'gene': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_genes = importance.head(n_features)['gene'].values
        
        print(f"  Top importance: {importance.iloc[0]['importance']:.4f}")
        print(f"  Selected {n_features} genes")
        
        return selected_genes, importance
    
    def recursive_feature_elimination(self, X, y, n_features=100):
        """RFE with logistic regression"""
        print(f"🔄 Recursive feature elimination...")
        
        # Use faster model for RFE
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features, step=100)
        
        rfe.fit(X, y)
        
        selected_genes = X.columns[rfe.support_]
        ranking = pd.DataFrame({
            'gene': X.columns,
            'ranking': rfe.ranking_
        }).sort_values('ranking')
        
        print(f"  Selected {len(selected_genes)} genes via RFE")
        
        return selected_genes, ranking
    
    def combine_feature_selections(self, expr_df, y, n_final=100):
        """Combine multiple feature selection methods"""
        print("="*60)
        print("🎯 COMBINING FEATURE SELECTION METHODS")
        print("="*60)
        
        all_selections = {}
        
        # 1. Univariate selection
        genes_uni, scores_uni = self.univariate_selection(expr_df, y, n_features=200)
        all_selections['univariate'] = set(genes_uni)
        
        # 2. Random Forest
        genes_rf, importance = self.random_forest_importance(expr_df, y, n_features=200)
        all_selections['random_forest'] = set(genes_rf)
        
        # 3. Differential expression (if binary)
        if len(np.unique(y)) == 2:
            de_results = self.differential_expression(expr_df, y)
            genes_de = de_results.head(200)['gene'].values
            all_selections['differential'] = set(genes_de)
        
        # Find consensus genes (appear in multiple methods)
        from collections import Counter
        gene_counts = Counter()
        for method_genes in all_selections.values():
            gene_counts.update(method_genes)
        
        # Genes selected by at least 2 methods
        consensus_genes = [gene for gene, count in gene_counts.items() if count >= 2]
        
        print(f"\n📊 Feature Selection Summary:")
        print(f"  Univariate: {len(all_selections['univariate'])} genes")
        print(f"  Random Forest: {len(all_selections['random_forest'])} genes")
        if 'differential' in all_selections:
            print(f"  Differential: {len(all_selections['differential'])} genes")
        print(f"  Consensus (≥2 methods): {len(consensus_genes)} genes")
        
        # Take top genes from consensus, then fill with top univariate
        final_genes = list(consensus_genes[:n_final])
        if len(final_genes) < n_final:
            # Add more from univariate selection
            additional = [g for g in genes_uni if g not in final_genes]
            final_genes.extend(additional[:n_final - len(final_genes)])
        
        self.top_genes = final_genes[:n_final]
        
        print(f"\n✅ Final selection: {len(self.top_genes)} genes")
        
        return self.top_genes
    
    def visualize_top_genes(self, expr_df, meta_df, top_n=20):
        """Create heatmap of top genes"""
        print("🎨 Creating visualization...")
        
        # Get top genes
        top_genes = self.top_genes[:top_n] if self.top_genes else expr_df.columns[:top_n]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        plot_data = expr_df[top_genes].T
        
        # Sort samples by diagnosis
        sample_order = meta_df.sort_values('diagnosis').index
        plot_data = plot_data[sample_order]
        
        # Create heatmap
        sns.heatmap(plot_data, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Expression (normalized)'},
                   xticklabels=False, yticklabels=True)
        
        # Add diagnosis color bar
        diagnosis_colors = {'Control': 'green', 'MCI': 'yellow', 
                          'MCI_converter': 'orange', 'AD': 'red'}
        colors = [diagnosis_colors[d] for d in meta_df.loc[sample_order, 'diagnosis']]
        
        # Add color bar at top
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, -1), 1, 0.5, color=color))
        
        ax.set_title(f'Top {top_n} Selected Genes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Samples (sorted by diagnosis)')
        ax.set_ylabel('Genes')
        
        plt.tight_layout()
        plt.savefig('results/top_genes_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()

# Run feature engineering
if __name__ == "__main__":
    # Load processed data
    expr = pd.read_csv('data/processed/expression_processed.csv.gz', 
                      compression='gzip', index_col=0)
    meta = pd.read_csv('data/processed/metadata_processed.csv', index_col=0)
    
    # Focus on MCI conversion
    mci_mask = meta['diagnosis'].isin(['MCI', 'MCI_converter'])
    X_mci = expr[mci_mask]
    y_mci = (meta[mci_mask]['diagnosis'] == 'MCI_converter').astype(int)
    meta_mci = meta[mci_mask]
    
    # Run feature engineering
    engineer = FeatureEngineer()
    selected_genes = engineer.combine_feature_selections(X_mci, y_mci, n_final=100)
    
    # Visualize
    engineer.visualize_top_genes(X_mci, meta_mci, top_n=30)
    
    # Save selected features
    X_mci_selected = X_mci[selected_genes]
    X_mci_selected.to_csv('data/processed/X_mci_top100_genes.csv.gz', compression='gzip')
    
    # Save gene list
    pd.DataFrame({'gene': selected_genes}).to_csv('data/processed/selected_genes.csv', index=False)
    
    print("\n💾 Selected features saved!")