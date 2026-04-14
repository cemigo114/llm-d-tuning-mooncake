#!/usr/bin/env bash
set -euo pipefail

# Deploy P/D with MultiConnector: NIXL (cross-node KV transfer) + OffloadingConnector (100GB CPU cache tier).
#
# This combines P/D disaggregation with CPU KV cache offloading for workloads
# with long input sequences (>8K tokens) where HBM pressure causes cache eviction.
#
# Prerequisites:
#   - Infrastructure already deployed (deploy-colocated.sh or just the infra chart)
#   - RDMA/IB available on nodes
#   - Sufficient CPU RAM (~400GB per pod requested)
#
# When to use this vs deploy-pd.sh:
#   - Input sequences > 8K tokens: MultiConnector wins (+43% throughput, -94% TTFT P90)
#   - Input sequences < 4K tokens: NIXL-only is faster (CPU offload adds overhead)

NAMESPACE="${NAMESPACE:-kpouget-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFESTS="$SCRIPT_DIR/../manifests"

echo "=== Deploying P/D with MultiConnector (NIXL + CPU 100GB offload) ==="

# Check infra
if ! helm status infra-gptoss -n "$NAMESPACE" &>/dev/null; then
  echo "ERROR: Infrastructure not deployed. Run deploy-colocated.sh first."
  exit 1
fi

# Tear down existing model server
for release in ms-pd ms-gptoss; do
  if helm status "$release" -n "$NAMESPACE" &>/dev/null; then
    echo "Removing existing model server ($release)..."
    helm uninstall "$release" -n "$NAMESPACE"
    sleep 10
  fi
done

# Deploy EPP if not present
if ! helm status gaie-pd -n "$NAMESPACE" &>/dev/null; then
  echo "[1/3] Installing P/D EPP..."
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
else
  echo "[1/3] P/D EPP already deployed, skipping."
fi

# Deploy MultiConnector model server
echo "[2/3] Installing MultiConnector model server (NIXL + CPU offload)..."
helm install ms-pd \
  llm-d-modelservice/llm-d-modelservice \
  -n "$NAMESPACE" \
  -f "$MANIFESTS/pd-multiconnector/ms-pd-multi-values.yaml"

# HTTPRoute
echo "[3/3] Applying HTTPRoute..."
kubectl apply -f "$MANIFESTS/pd/httproute-pd.yaml" -n "$NAMESPACE"

echo ""
echo "=== MultiConnector deployment initiated ==="
echo ""
echo "vLLM KV transfer config:"
echo '  MultiConnector → [NixlConnector (P/D RDMA), OffloadingConnector (100GB CPU)]'
echo ""
echo "Monitor: kubectl get pods -n $NAMESPACE -w"
echo "Wait:    kubectl wait --for=condition=Ready pod -l llm-d.ai/role -n $NAMESPACE --timeout=600s"
echo ""
echo "Best for: long input sequences (>8K tokens) where CPU tier provides cache hits"
echo "Not for:  short sequences (<4K tokens) where CPU offload overhead dominates"
