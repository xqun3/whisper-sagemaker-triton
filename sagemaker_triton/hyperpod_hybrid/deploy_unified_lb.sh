#!/bin/bash

set -e

echo "=== Whisper Triton Unified LoadBalancer Deployment ==="

# Step 1: Create ConfigMap
echo "Creating ConfigMap for scripts..."
kubectl create configmap whisper-scripts --from-file=../sagemaker_triton/model_data/ --dry-run=client -o yaml | kubectl apply -f -

# Step 2: Deploy unified LoadBalancer setup
echo "Deploying Whisper Triton with unified LoadBalancer..."
kubectl apply -f whisper-triton-unified-lb.yaml

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
echo "✅ Unified LoadBalancer deployment completed!"
echo ""
echo "Single entry point:"
echo "Unified: $(kubectl get service whisper-triton-unified-nlb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'):8080"
echo ""
echo "This LoadBalancer will distribute traffic across both G6E and G5 instances."
