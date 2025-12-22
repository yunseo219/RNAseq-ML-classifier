import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_manager_v2 import RobustDataManager

def test_robust_download():
    print("="*60)
    print("🧬 Testing Robust Download Method")
    print("="*60)
    
    manager = RobustDataManager(
        use_aws=True,
        bucket_name='ad-rnaseq-prediction-data'
    )
    
    # Try the alternative download
    expr, meta = manager.download_geo_alternative('GSE63061')
    
    if expr is not None:
        print(f"\n✅ Success!")
        print(f"Expression shape: {expr.shape}")
        print(f"Metadata shape: {meta.shape}")
        
        # Show some basic info
        print(f"\nFirst 5 samples:")
        print(expr.iloc[:5, :5])
        
        print(f"\nMetadata columns:")
        print(meta.columns.tolist()[:10])
        
        if 'diagnosis' in meta.columns:
            print(f"\nDiagnosis distribution:")
            print(meta['diagnosis'].value_counts())
    else:
        print("❌ Download failed")

if __name__ == "__main__":
    test_robust_download()