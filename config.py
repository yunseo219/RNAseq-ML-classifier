# Configuration for AD RNA-seq Prediction Project

# AWS Settings
AWS_BUCKET = 'ad-rnaseq-prediction-data'
AWS_REGION = 'us-west-2'
USE_AWS = True

# Datasets to download
DATASETS = {
    'primary': 'GSE63061',
    'validation': 'GSE63060', 
    'additional': 'GSE140829'
}

# Local paths (Windows-safe)
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')

# Analysis parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5