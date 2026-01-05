import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")



s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def download_file_from_s3(s3_path: str) -> bytes:
    """
    Download a file from S3 and return its content as bytes.

    s3_path format: "bucket_name/path/to/file.csv"
    """
    try:
        bucket, *key_parts = s3_path.split("/")
        key = "/".join(key_parts)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except ClientError as e:
        raise Exception(f"Failed to download S3 file {s3_path}: {e}")
