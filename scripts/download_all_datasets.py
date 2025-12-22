import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_manager import SmartDataManager
import pandas as pd
import time

def download_all():
    """Download all three datasets for the project"""
    
    manager = SmartDataManager(
        use_aws=True,
        bucket_name='ad-rnaseq-prediction-data'
    )
    
    datasets = {
        'GSE63061': 'Primary AD blood RNA-seq dataset',
        'GSE63060': 'Validation dataset from ADNI',
        'GSE140829': 'Additional progression dataset'
    }
    
    results = {}
    
    for geo_id, description in datasets.items():
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {geo_id}")
        print(f"📝 Description: {description}")
        print(f"{'='*60}")
        
        try:
            start = time.time()
            
            # Keep primary dataset local, others only on S3
            keep_local = (geo_id == 'GSE63061')
            
            expr, meta = manager.download_geo_dataset(geo_id, keep_local=keep_local)
            
            elapsed = time.time() - start
            
            results[geo_id] = {
                'status': 'success',
                'samples': expr.shape[0],
                'genes': expr.shape[1],
                'time_minutes': elapsed/60
            }
            
            print(f"✅ Success: {expr.shape[0]} samples, {expr.shape[1]} genes")
            print(f"⏱️ Time: {elapsed/60:.1f} minutes")
            
        except Exception as e:
            results[geo_id] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ Failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    for geo_id, info in results.items():
        if info['status'] == 'success':
            print(f"✅ {geo_id}: {info['samples']} samples, {info['time_minutes']:.1f} min")
        else:
            print(f"❌ {geo_id}: {info['error'][:50]}...")
    
    # Save summary
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv('data/download_summary.csv')
    print(f"\n📁 Summary saved to data/download_summary.csv")

if __name__ == "__main__":
    download_all()