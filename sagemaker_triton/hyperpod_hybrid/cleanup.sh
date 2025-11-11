#!/bin/bash

echo "=== Cleaning up Whisper Triton Deployment ==="

# Delete everything from unified YAML
kubectl delete -f whisper-triton-complete.yaml --ignore-not-found=true

# Delete ConfigMap
kubectl delete configmap whisper-scripts --ignore-not-found=true

echo "✅ Cleanup completed!"
echo "Note: AWS LoadBalancers will be automatically deleted when services are removed."
