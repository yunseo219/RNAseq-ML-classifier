import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_manager import SmartDataManager
import time

def test_download():
    print("="*60)
    print("🧬 AD RNA-seq Data Download Test")
    print("="*60)
    
    # Initialize manager with your working AWS bucket
    manager = SmartDataManager(
        use_aws=True,
        bucket_name='ad-rnaseq-prediction-data'
    )
    
    # Test with the primary dataset
    print("\n📊 Downloading GSE63061 (Primary AD Dataset)...")
    print("This may take 5-10 minutes for first download...")
    
    start_time = time.time()
    
    try:
        expr, meta = manager.download_geo_dataset(
            'GSE63061', 
            keep_local=True  # Keep a local copy for quick access
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Download completed in {elapsed/60:.1f} minutes")
        print(f"\n📈 Dataset Statistics:")
        print(f"  - Samples: {expr.shape[0]}")
        print(f"  - Genes: {expr.shape[1]}")
        print(f"  - Metadata columns: {len(meta.columns)}")
        
        # Show diagnosis distribution if available
        for col in meta.columns:
            if 'disease' in col.lower() or 'diagnosis' in col.lower():
                print(f"\n🏥 Diagnoses in {col}:")
                print(meta[col].value_counts())
                break
                
        return expr, meta
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    expr, meta = test_download()