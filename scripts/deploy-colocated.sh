#!/usr/bin/env bash
set -euo pipefail

# Deploy llm-d with co-located (non-disaggregated) gpt-oss-120b
# Topology: 4 vLLM replicas × TP=4 = 16 GPUs
#
# Prerequisites:
#   - KUBECONFIG set to your cluster
#   - NAMESPACE set (default: kpouget-dev)
#   - llm-d-hf-token secret created in the namespace
#   - Helm repos: llm-d-infra, llm-d-modelservice

NAMESPACE="${NAMESPACE:-kpouget-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFESTS="$SCRIPT_DIR/../manifests"

echo "=== Deploying llm-d co-located stack to $NAMESPACE ==="

# 1. Infrastructure (Gateway)
echo "[1/4] Installing infrastructure..."
helm install infra-gptoss llm-d-infra/llm-d-infra \
  -n "$NAMESPACE" \
  --set gateway.provider=istio \
  --set 'gateway.gatewayParameters.resources.limits.cpu=8' \
  --set 'gateway.gatewayParameters.resources.limits.memory=8Gi' \
  --set 'gateway.gatewayParameters.resources.requests.cpu=2' \
  --set 'gateway.gatewayParameters.resources.requests.memory=2Gi' \
  --set 'gateway.gatewayParameters.istio.accessLogging=false'

# 2. InferencePool + EPP
echo "[2/4] Installing EPP inference scheduler..."
helm install gaie-gptoss \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --version v1.4.0 \
  -n "$NAMESPACE" \
  -f "$MANIFESTS/colocated/gaie-values.yaml" \
  --set 'provider.name=istio' \
  --set "provider.istio.destinationRule.host=gaie-gptoss-epp.$NAMESPACE.svc.cluster.local" \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.http1MaxPendingRequests=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.maxRequestsPerConnection=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.http2MaxRequests=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.tcp.maxConnections=256000'

# 3. Model server (4 × TP=4 vLLM replicas)
echo "[3/4] Installing model server (gpt-oss-120b, 4×TP4)..."
helm install ms-gptoss \
  llm-d-modelservice/llm-d-modelservice \
  -n "$NAMESPACE" \
  -f "$MANIFESTS/colocated/ms-values.yaml"

# 4. HTTPRoute
echo "[4/4] Applying HTTPRoute..."
kubectl apply -f "$MANIFESTS/colocated/httproute.yaml" -n "$NAMESPACE"

echo ""
echo "=== Deployment initiated ==="
echo "Monitor with: kubectl get pods -n $NAMESPACE -w"
echo "Test with:    kubectl port-forward -n $NAMESPACE svc/infra-gptoss-inference-gateway-istio 8080:80"
echo "              curl http://localhost:8080/v1/models"
