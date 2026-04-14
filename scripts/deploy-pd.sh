#!/usr/bin/env bash
set -euo pipefail

# Deploy llm-d with P/D disaggregation for gpt-oss-120b
# Topology: 2 prefill (TP=4) + 2 decode (TP=4) = 16 GPUs
# KV cache transfer via NIXL over RDMA/InfiniBand
#
# Prerequisites:
#   - KUBECONFIG set to your cluster
#   - NAMESPACE set (default: kpouget-dev)
#   - llm-d-hf-token secret created in the namespace
#   - RDMA/IB available on nodes (rdma/ib resource)
#   - Infrastructure already deployed (run deploy-colocated.sh first,
#     or just the infra chart)

NAMESPACE="${NAMESPACE:-kpouget-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFESTS="$SCRIPT_DIR/../manifests"

echo "=== Deploying llm-d P/D disaggregation to $NAMESPACE ==="

# Check if infra is already deployed
if ! helm status infra-gptoss -n "$NAMESPACE" &>/dev/null; then
  echo "ERROR: Infrastructure not deployed. Run deploy-colocated.sh first or install infra chart."
  exit 1
fi

# Tear down co-located if present
if helm status ms-gptoss -n "$NAMESPACE" &>/dev/null; then
  echo "[0/3] Removing co-located model server..."
  helm uninstall ms-gptoss -n "$NAMESPACE"
fi
if helm status gaie-gptoss -n "$NAMESPACE" &>/dev/null; then
  echo "[0/3] Removing co-located EPP..."
  helm uninstall gaie-gptoss -n "$NAMESPACE"
  echo "Waiting for pods to terminate..."
  sleep 10
fi

# 1. InferencePool + EPP with P/D config
echo "[1/3] Installing P/D EPP inference scheduler..."
helm install gaie-pd \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --version v1.4.0 \
  -n "$NAMESPACE" \
  -f "$MANIFESTS/pd/gaie-pd-values.yaml" \
  --set 'provider.name=istio' \
  --set "provider.istio.destinationRule.host=gaie-pd-epp.$NAMESPACE.svc.cluster.local" \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.http1MaxPendingRequests=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.maxRequestsPerConnection=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.http2MaxRequests=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.tcp.maxConnections=256000'

# 2. Model server (2P + 2D with NIXL routing sidecar)
echo "[2/3] Installing P/D model server (2 prefill + 2 decode, TP=4)..."
helm install ms-pd \
  llm-d-modelservice/llm-d-modelservice \
  -n "$NAMESPACE" \
  -f "$MANIFESTS/pd/ms-pd-values.yaml"

# 3. HTTPRoute pointing to P/D pool
echo "[3/3] Applying P/D HTTPRoute..."
kubectl apply -f "$MANIFESTS/pd/httproute-pd.yaml" -n "$NAMESPACE"

echo ""
echo "=== P/D deployment initiated ==="
echo "Model loading takes ~5-7 minutes on H200."
echo "Monitor with: kubectl get pods -n $NAMESPACE -w"
echo "Wait with:    kubectl wait --for=condition=Ready pod -l 'llm-d.ai/role' -n $NAMESPACE --timeout=600s"
echo "Test with:    kubectl port-forward -n $NAMESPACE svc/infra-gptoss-inference-gateway-istio 8080:80"
