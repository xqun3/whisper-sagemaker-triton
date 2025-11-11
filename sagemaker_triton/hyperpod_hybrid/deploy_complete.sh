#!/bin/bash

set -e

echo "=== Whisper Triton HyperPod EKS Deployment ==="

# Step 1: Create ConfigMap
echo "Creating ConfigMap for scripts..."
kubectl create configmap whisper-scripts --from-file=../sagemaker_triton/model_data/ --dry-run=client -o yaml | kubectl apply -f -

# Step 2: Deploy everything (deployments + services)
echo "Deploying Whisper Triton (deployments + services)..."
kubectl apply -f whisper-triton-complete.yaml

# Step 3: Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/whisper-triton-g6e
kubectl wait --for=condition=available --timeout=300s deployment/whisper-triton-g5

# Step 4: Show status
echo "=== Deployment Status ==="
kubectl get pods -o wide
echo ""
kubectl get services | grep whisper

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "Service endpoints:"
echo "G6E: $(kubectl get service whisper-triton-g6e-nlb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'):8080"
echo "G5:  $(kubectl get service whisper-triton-g5-nlb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'):8080"
