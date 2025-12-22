import pandas as pd
from pathlib import Path

def summarize_project_data():
    """Generate summary report of all data"""
    
    print("="*60)
    print("📊 AD RNA-seq Prediction Project - Data Summary")
    print("="*60)
    
    # Check processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        files = list(processed_dir.glob("*.csv*"))
        print(f"\n✅ Processed datasets: {len(files)} files")
        for f in files:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"   - {f.name}: {size_mb:.1f} MB")
    
    # Load and summarize
    try:
        y_mci = pd.read_csv("data/processed/y_mci_conversion.csv", index_col=0)
        print(f"\n🎯 MCI Conversion Target:")
        print(f"   Total samples: {len(y_mci)}")
        print(f"   Converters: {y_mci['will_convert'].sum()}")
        print(f"   Conversion rate: {y_mci['will_convert'].mean()*100:.1f}%")
    except:
        pass
    
    print("\n📅 Next Steps:")
    print("   1. Feature engineering (Week 3)")
    print("   2. Model development (Week 4)")  
    print("   3. Validation (Week 5)")
    print("   4. Deployment (Week 6)")

if __name__ == "__main__":
    summarize_project_data()