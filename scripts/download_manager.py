import os
import sys
import GEOparse
import boto3
import pandas as pd
from pathlib import Path
import shutil

class SmartDataManager:
    def __init__(self, use_aws=True, bucket_name='ad-rnaseq-prediction-data'):
        self.use_aws = use_aws
        self.bucket_name = bucket_name
        
        if use_aws:
            try:
                self.s3 = boto3.client('s3')
                print("✓ AWS S3 configured")
            except Exception as e:
                print(f"⚠ AWS not configured: {e}")
                print("Run 'aws configure' first")
                self.use_aws = False
        else:
            print("⚠ Local storage mode")
    
    def download_geo_dataset(self, geo_id, keep_local=False):
        """
        Smart download: Goes to AWS if configured, otherwise local
        """
        print(f"\n📥 Downloading {geo_id}...")
        
        # Check if already exists in S3
        if self.use_aws and self.check_s3_exists(geo_id):
            print(f"✓ {geo_id} already in S3, skipping download")
            return self.load_from_s3(geo_id)
        
        # Create temp directory for download (Windows-safe path)
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / "temp" / geo_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download from GEO
            print(f"Downloading from GEO...")
            gse = GEOparse.get_GEO(geo_id, destdir=str(temp_dir))
            
            # Extract data
            expression = gse.get_expression_data()
            metadata = gse.phenotype_data
            
            print(f"✓ Downloaded: {expression.shape[0]} samples, {expression.shape[1]} genes")
            
            # Save as compressed files
            expr_file = temp_dir / f"{geo_id}_expression.csv.gz"
            meta_file = temp_dir / f"{geo_id}_metadata.csv"
            
            expression.to_csv(expr_file, compression='gzip')
            metadata.to_csv(meta_file)
            
            if self.use_aws:
                # Upload to S3
                print(f"\n☁️ Uploading to AWS S3...")
                self.upload_to_s3(expr_file, f"raw/{geo_id}/{geo_id}_expression.csv.gz")
                self.upload_to_s3(meta_file, f"raw/{geo_id}/{geo_id}_metadata.csv")
                print(f"✓ Uploaded to S3: s3://{self.bucket_name}/raw/{geo_id}/")
                
                if not keep_local:
                    shutil.rmtree(temp_dir)
                    print(f"🧹 Cleaned up local temp files")
            else:
                # Move to local data directory
                final_dir = project_root / "data" / "raw" / geo_id
                final_dir.mkdir(parents=True, exist_ok=True)
                
                shutil.move(str(expr_file), str(final_dir / expr_file.name))
                shutil.move(str(meta_file), str(final_dir / meta_file.name))
                
                # Clean temp
                shutil.rmtree(temp_dir)
                print(f"✓ Saved locally to data/raw/{geo_id}/")
            
            return expression, metadata
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def check_s3_exists(self, geo_id):
        """Check if dataset already exists in S3"""
        try:
            self.s3.head_object(
                Bucket=self.bucket_name,
                Key=f"raw/{geo_id}/{geo_id}_expression.csv.gz"
            )
            return True
        except:
            return False
    
    def upload_to_s3(self, local_file, s3_key):
        """Upload file to S3"""
        self.s3.upload_file(
            str(local_file),
            self.bucket_name,
            s3_key
        )
    
    def load_from_s3(self, geo_id):
        """Load dataset from S3"""
        print(f"📥 Loading {geo_id} from S3...")
        
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / "temp" / f"{geo_id}_load"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        expr_file = temp_dir / f"{geo_id}_expression.csv.gz"
        meta_file = temp_dir / f"{geo_id}_metadata.csv"
        
        self.s3.download_file(
            self.bucket_name,
            f"raw/{geo_id}/{geo_id}_expression.csv.gz",
            str(expr_file)
        )
        
        self.s3.download_file(
            self.bucket_name,
            f"raw/{geo_id}/{geo_id}_metadata.csv",
            str(meta_file)
        )
        
        expression = pd.read_csv(expr_file, compression='gzip', index_col=0)
        metadata = pd.read_csv(meta_file, index_col=0)
        
        shutil.rmtree(temp_dir)
        
        return expression, metadata