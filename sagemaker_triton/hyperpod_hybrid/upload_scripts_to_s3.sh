#!/bin/bash

set -e

echo "=== Uploading Scripts to S3 Bucket ==="

S3_BUCKET="triton-models-xq/deployment_codes"
SCRIPTS_DIR="../sagemaker_triton/model_data"

echo "Uploading scripts to s3://$S3_BUCKET/"
echo "Source directory: $SCRIPTS_DIR"
echo ""

# Upload individual script files
echo "Uploading run_server.py..."
aws s3 cp "$SCRIPTS_DIR/run_server.py" "s3://$S3_BUCKET/"

echo "Uploading whisper_api.py..."
aws s3 cp "$SCRIPTS_DIR/whisper_api.py" "s3://$S3_BUCKET/"

echo "Uploading deploy_config.sh..."
aws s3 cp "$SCRIPTS_DIR/deploy_config.sh" "s3://$S3_BUCKET/"

# Optional: Upload all files
# echo "Uploading all script files..."
# aws s3 cp "$SCRIPTS_DIR/" "s3://$S3_BUCKET/" --recursive --exclude="*.pyc" --exclude="__pycache__/*"

# Verify upload
echo ""
echo "Verifying uploaded files:"
aws s3 ls "s3://$S3_BUCKET/" --recursive | grep -E "\.(py|sh)$"

echo ""
echo "✅ Scripts successfully uploaded to S3!"
echo ""
echo "S3 bucket structure should now be:"
echo "  s3://triton-models-xq/"
echo "  ├── test_turbo_g6e/           # G6E model files"
echo "  ├── test_turbo/               # G5 model files"
echo "  └── deployment_codes/         # Deployment scripts"
echo "      ├── run_server.py         # API server script"
echo "      ├── whisper_api.py        # Whisper API implementation"
echo "      └── deploy_config.sh      # Config script"
echo ""
echo "You can now run: ./deploy_pv_scripts.sh"
