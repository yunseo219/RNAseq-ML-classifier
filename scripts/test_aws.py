import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def test_aws_setup():
    try:
        # Test credentials
        s3 = boto3.client('s3')
        
        # List buckets
        response = s3.list_buckets()
        print("✅ AWS credentials valid!")
        print(f"Found {len(response['Buckets'])} buckets:")
        for bucket in response['Buckets']:
            print(f"  - {bucket['Name']}")
        
        # Test specific bucket
        bucket_name = 'ad-rnaseq-prediction-data'
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"\n✅ Can access bucket: {bucket_name}")
        except ClientError:
            print(f"\n❌ Cannot access bucket: {bucket_name}")
            print("   Check if bucket exists and you have permissions")
            
    except NoCredentialsError:
        print("❌ No AWS credentials found!")
        print("   Run: aws configure")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_aws_setup()