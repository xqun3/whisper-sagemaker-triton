# Whisper Triton HyperPod EKS Deployment

This directory contains the successful deployment configuration for Whisper Triton on Amazon EKS HyperPod.

## Model Compilation and Upload
1. Compile models - different instance types require compilation on corresponding GPU machines
2. Upload models to S3
3. Bind the S3 path as PV in the cluster

## S3 Bucket PV/PVC Binding

### Create PersistentVolume and PersistentVolumeClaim for S3 bucket:

```bash
# Apply PV configuration
kubectl apply -f pv-triton-models.yaml

# Apply PVC configuration  
kubectl apply -f pvc-triton-models.yaml

# Verify binding
kubectl get pv,pvc
```

The configurations use AWS S3 CSI driver to mount `triton-models-xq` S3 bucket:
- **PV**: `pv-triton-models` (1200Gi, ReadWriteMany)
- **PVC**: `triton-models` (bound to PV)
- **Mount options**: `allow-delete`, `region us-east-1`

## Deployment Options

### Option 1: ConfigMap Scripts (Current)
- `whisper-triton-unified-lb.yaml` - Uses ConfigMap for script distribution
- `deploy_unified_lb.sh` - Deploy with ConfigMap

### Option 2: PV Scripts (Simplified)
- `whisper-triton-pv-scripts.yaml` - Scripts stored directly in S3/PV
- `upload_scripts_to_s3.sh` - Upload scripts to S3 bucket
- `deploy_pv_scripts.sh` - Deploy with PV-based scripts

### Option 3: Separate LoadBalancers
- `whisper-triton-complete.yaml` - Two separate LoadBalancers
- `deploy_complete.sh` - Deploy with separate endpoints

## Quick Deployment

### PV Scripts (Recommended - Simplest):
```bash
# 1. Upload scripts to S3
./upload_scripts_to_s3.sh

# 2. Deploy
./deploy_pv_scripts.sh
```

### ConfigMap Scripts:
```bash
./deploy_unified_lb.sh
```

### Cleanup:
```bash
./cleanup.sh
```

## Configuration

- **G6E Instance**: `ml.g6e.2xlarge` with model path `test_turbo_g6e`
- **G5 Instance**: `ml.g5.2xlarge` with model path `test_turbo`
- **S3 Bucket**: `triton-models-xq` containing models and scripts
- **Ports**: 
  - Triton Server: 10086
  - API Server: 8080

## S3 Bucket Structure (PV Scripts)

```
s3://triton-models-xq/
├── test_turbo_g6e/           # G6E model files
├── test_turbo/               # G5 model files
└── deployment_codes/         # Deployment scripts
    ├── run_server.py         # API server script
    └── whisper_api.py        # Whisper API implementation
```

## Service Endpoints

### Unified LoadBalancer (Single Entry Point)
- **Unified NLB**: `whisper-triton-unified-nlb`
  - Current: `k8s-default-whispert-72e1749839-7dc3f5766c221754.elb.us-east-2.amazonaws.com:8080`
  - IP: `18.219.99.188:8080`

## Testing

```bash
# Test unified endpoint
python3 test_unified_lb.py

# Manual test
curl http://18.219.99.188:8080/ping

# real audio test
curl -X POST http://k8s-default-whispert-eb4eb229b7-03296d4e2539c9bf.elb.us-east-2.amazonaws.com:8080/invocations -H "Content-Type: application/json" -d "{\"audio_data\": \"$(base64 -w 0 test.wav)\", \"whisper_prompt\": \"\"}" | jq .

# Python API test
python3 -c "
import requests
import base64

with open('audio.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')

url = 'http://k8s-default-whispert-eb4eb229b7-03296d4e2539c9bf.elb.us-east-2.amazonaws.com:8080/invocations'
payload = {'audio_data': audio_b64, 'whisper_prompt': ''}

response = requests.post(url, json=payload)
result = response.json()

print(f\"Status: {response.status_code}\")
print(f\"Result: {result}\")
```

## Advantages of PV Scripts

1. **No ConfigMap needed** - Scripts stored directly with models
2. **Simpler deployment** - Fewer Kubernetes resources
3. **Easier updates** - Just upload to S3, restart pods
4. **Version control** - S3 versioning for scripts
5. **Shared storage** - Scripts available to all pods

✅ **Status**: All deployment options fully operational!
