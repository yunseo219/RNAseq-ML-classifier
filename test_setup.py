import os
import sys
from pathlib import Path

def test_setup():
    print("🧪 Testing setup on Windows...")
    print(f"Current directory: {os.getcwd()}")
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Test imports
    packages = ['boto3', 'GEOparse', 'pandas', 'numpy']
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed - run: pip install {package}")
    
    # Test AWS
    try:
        import boto3
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        print("✅ AWS configured")
    except:
        print("❌ AWS not configured - run: aws configure")
    
    # Test project structure
    required_dirs = ['scripts', 'notebooks', 'data', 'data\\raw', 'data\\processed', 'temp']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}\\ exists")
        else:
            print(f"⚠️ Creating {dir_name}\\...")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n🎉 Setup complete!")

if __name__ == "__main__":
    test_setup()