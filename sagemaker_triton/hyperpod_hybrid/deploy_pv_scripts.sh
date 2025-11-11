#!/bin/bash

set -e

echo "=== Whisper Triton PV Scripts Deployment ==="

echo "This deployment uses scripts directly from S3 bucket 'triton-models-xq'"
echo "Make sure you have run: ./upload_scripts_to_s3.sh first"
echo ""

read -p "Have you uploaded scripts to S3? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please run: ./upload_scripts_to_s3.sh first"
    exit 1
fi

# Deploy PV-based setup (no ConfigMap needed)
echo "Deploying Whisper Triton with PV-based scripts..."
kubectl apply -f whisper-triton-pv-scripts.yaml

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/whisper-triton-g6e
kubectl wait --for=condition=available --timeout=300s deployment/whisper-triton-g5

# Show status
echo "=== Deployment Status ==="
kubectl get pods -o wide
echo ""
kubectl get services | grep whisper

echo ""
echo "✅ PV Scripts deployment completed!"
echo ""
echo "Single entry point:"
echo "Unified: $(kubectl get service whisper-triton-unified-nlb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'):8080"
