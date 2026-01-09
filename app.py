import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AD Prediction from RNA-seq",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #666;
    text-align: center;
    margin-bottom: 3rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧬 Alzheimer\'s Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Prediction from Blood RNA-seq Data</p>', unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_info = pd.read_csv('models/best_model_info.csv')
    model_type = model_info.iloc[0]['model_type']
    model = joblib.load(f'models/best_model_{model_type}.pkl')
    return model, model_info

@st.cache_data
def load_data():
    """Load processed data"""
    expr = pd.read_csv('data/processed/expression_processed.csv.gz', compression='gzip', index_col=0)
    meta = pd.read_csv('data/processed/metadata_processed.csv', index_col=0)
    return expr, meta

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=RNA-seq+ML", use_column_width=True)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔬 Data Explorer", "🤖 Make Prediction", "📊 Model Performance", "📚 About"]
    )
    
    st.markdown("---")
    st.markdown("### Project Info")
    st.info("""
    **Dataset**: GSE63061  
    **Samples**: 389  
    **Features**: 5000 genes  
    **Target**: MCI→AD conversion
    """)

# Load resources
try:
    model, model_info = load_model()
    expr, meta = load_data()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Main content based on page selection
if page == "🏠 Home":
    st.header("Welcome to the AD Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(meta))
    with col2:
        mci_count = (meta['diagnosis'].isin(['MCI', 'MCI_converter'])).sum()
        st.metric("MCI Samples", mci_count)
    with col3:
        if model_loaded:
            auc_score = model_info.iloc[0]['auc']
            st.metric("Model AUC", f"{auc_score:.3f}")
    
    st.markdown("---")
    
    # Overview
    st.subheader("🎯 Project Overview")
    st.markdown("""
    This application uses machine learning to predict Alzheimer's Disease progression from blood-based 
    RNA sequencing data. The model specifically focuses on predicting which patients with 
    Mild Cognitive Impairment (MCI) will convert to Alzheimer's Disease.
    
    ### Key Features:
    - **Non-invasive**: Uses blood samples instead of brain tissue
    - **Early detection**: Identifies at-risk MCI patients
    - **Machine Learning**: Advanced algorithms for accurate prediction
    - **5000 gene features**: Selected through rigorous feature engineering
    """)
    
    # Show sample distribution
    st.subheader("📊 Dataset Distribution")
    
    fig_dist = px.pie(
        values=meta['diagnosis'].value_counts().values,
        names=meta['diagnosis'].value_counts().index,
        title="Sample Distribution by Diagnosis",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_dist, use_container_width=True)

elif page == "🔬 Data Explorer":
    st.header("Data Explorer")
    
    # Data overview
    st.subheader("Expression Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Expression Matrix Shape:**")
        st.info(f"{expr.shape[0]} samples × {expr.shape[1]} genes")
    with col2:
        st.write("**Data Statistics:**")
        st.info(f"Mean: {expr.mean().mean():.2f}, Std: {expr.std().mean():.2f}")
    
    # Show sample data
    if st.checkbox("Show sample expression data"):
        st.dataframe(expr.iloc[:10, :10])
    
    # PCA visualization
    st.subheader("PCA Visualization")
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(expr)
    
    pca_df = pd.DataFrame(
        pca_coords, 
        columns=['PC1', 'PC2'],
        index=expr.index
    )
    pca_df['Diagnosis'] = meta['diagnosis']
    
    fig_pca = px.scatter(
        pca_df, x='PC1', y='PC2', color='Diagnosis',
        title=f"PCA Plot (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})",
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
        }
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Gene expression heatmap
    st.subheader("Top Variable Genes Heatmap")
    
    # Select top variable genes
    gene_vars = expr.var()
    top_genes = gene_vars.nlargest(50).index
    
    # Create heatmap data
    heatmap_data = expr[top_genes].T
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu',
        zmid=0
    ))
    fig_heat.update_layout(
        title="Top 50 Variable Genes",
        xaxis_title="Samples",
        yaxis_title="Genes",
        height=800
    )
    st.plotly_chart(fig_heat, use_container_width=True)

elif page == "🤖 Make Prediction":
    st.header("Make Prediction")
    
    if not model_loaded:
        st.error("Model not loaded. Please check model files.")
    else:
        st.markdown("""
        ### Predict MCI to AD Conversion
        
        Select an MCI sample or upload new data to predict conversion probability.
        """)
        
        # Option 1: Select existing sample
        st.subheader("Option 1: Test on Existing Sample")
        
        mci_samples = meta[meta['diagnosis'].isin(['MCI', 'MCI_converter'])].index
        selected_sample = st.selectbox("Select an MCI sample:", mci_samples)
        
        if st.button("Predict for Selected Sample"):
            # Get sample data
            sample_data = expr.loc[[selected_sample]]
            true_label = meta.loc[selected_sample, 'diagnosis']
            
            # Make prediction
            try:
                # Handle scaling if model is Neural Network
                if 'Neural' in model_info.iloc[0]['model_type']:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    # Fit on training data (approximation)
                    scaler.fit(expr)
                    sample_scaled = scaler.transform(sample_data)
                    prediction_proba = model.predict_proba(sample_scaled)[0]
                else:
                    prediction_proba = model.predict_proba(sample_data)[0]
                
                prediction = "Converter" if prediction_proba[1] > 0.5 else "Non-converter"
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sample ID", selected_sample)
                with col2:
                    st.metric("True Label", true_label)
                with col3:
                    st.metric("Prediction", prediction)
                
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba[1] * 100,
                    title={'text': "Conversion Probability (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Interpretation
                if prediction_proba[1] > 0.7:
                    st.error("⚠️ High risk of conversion to AD")
                elif prediction_proba[1] > 0.5:
                    st.warning("⚡ Moderate risk of conversion")
                else:
                    st.success("✅ Low risk of conversion")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
        
        # Option 2: Upload new data
        st.subheader("Option 2: Upload New Data")
        st.info("Upload a CSV file with gene expression values (same format as training data)")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.warning("File upload prediction not implemented in this demo")

elif page == "📊 Model Performance":
    st.header("Model Performance")
    
    # Load prediction results
    pred_dir = Path('results/predictions')
    if pred_dir.exists():
        pred_files = list(pred_dir.glob('*.csv'))
        
        if pred_files:
            # Model comparison
            st.subheader("Model Comparison")
            
            results_data = []
            for file in pred_files:
                df = pd.read_csv(file)
                model_name = file.stem.replace('_predictions', '').replace('_', ' ')
                
                acc = (df['true_label'] == df['predicted_label']).mean()
                
                # Calculate sensitivity and specificity
                tp = ((df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
                tn = ((df['true_label'] == 0) & (df['predicted_label'] == 0)).sum()
                fp = ((df['true_label'] == 0) & (df['predicted_label'] == 1)).sum()
                fn = ((df['true_label'] == 1) & (df['predicted_label'] == 0)).sum()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                results_data.append({
                    'Model': model_name,
                    'Accuracy': acc,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Bar chart
            fig_bar = px.bar(
                results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric',
                title='Model Performance Comparison',
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            
            cols = st.columns(min(3, len(pred_files)))
            for idx, file in enumerate(pred_files[:3]):
                df = pd.read_csv(file)
                model_name = file.stem.replace('_predictions', '')
                
                tp = ((df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
                tn = ((df['true_label'] == 0) & (df['predicted_label'] == 0)).sum()
                fp = ((df['true_label'] == 0) & (df['predicted_label'] == 1)).sum()
                fn = ((df['true_label'] == 1) & (df['predicted_label'] == 0)).sum()
                
                cm = [[tn, fp], [fn, tp]]
                
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Non-converter', 'Converter'],
                    y=['Non-converter', 'Converter'],
                    title=model_name,
                    text_auto=True
                )
                cols[idx].plotly_chart(fig_cm, use_container_width=True)
    
    # Best model details
    if model_loaded:
        st.subheader("Best Model Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Type:**")
            st.info(model_info.iloc[0]['model_type'])
        with col2:
            st.write("**AUC Score:**")
            st.info(f"{model_info.iloc[0]['auc']:.3f}")

elif page == "📚 About":
    st.header("About This Project")
    
    st.markdown("""
    ## 🧬 Alzheimer's Disease Prediction from Blood RNA-seq
    
    ### Project Overview
    This project develops machine learning models to predict Alzheimer's Disease (AD) progression 
    using blood-based RNA sequencing data. The focus is on identifying which patients with 
    Mild Cognitive Impairment (MCI) will convert to AD.
    
    ### Technical Stack
    - **Data Processing**: Python, Pandas, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost, Neural Networks
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Web Framework**: Streamlit
    - **Cloud Storage**: AWS S3
    
    ### Dataset
    - **Source**: GEO Database (GSE63061)
    - **Samples**: 389 (145 AD, 134 Control, 80 MCI, 30 MCI converters)
    - **Features**: 5000 selected genes from 20,000+ initial genes
    
    ### Pipeline
    1. **Data Acquisition**: Downloaded from GEO, stored in AWS S3
    2. **Preprocessing**: Quality control, normalization, outlier detection
    3. **Feature Engineering**: Differential expression, variance-based selection
    4. **Model Training**: Multiple algorithms with hyperparameter tuning
    5. **Evaluation**: Cross-validation, AUC-ROC analysis
    6. **Deployment**: Streamlit web application
    
    ### Key Achievements
    - ✅ Identified 30 MCI converters for prediction
    - ✅ Reduced dimensionality from 20,000 to 5,000 genes
    - ✅ Achieved AUC > 0.7 for conversion prediction
    - ✅ Built interactive web application
    
    ### Author
    **Sebrina**  
    MS Computer Science Student  
    Transitioning to Data Science/Bioinformatics
    
    ### Repository
    [GitHub: ad-rnaseq-prediction](https://github.com/yunseo219/RNAseq-ML-classifier)
    
    ---
    
    *This project demonstrates end-to-end ML pipeline development for translational bioinformatics.*
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🧬 AD RNA-seq Prediction | Built with Streamlit | 2024
    </div>
    """,
    unsafe_allow_html=True
)