#!/usr/bin/env bash
set -euo pipefail

# Hot-swap the EPP deployment to use the adaptive scorer image.
# Requires P/D deployment already running (deploy-pd.sh).
#
# Usage:
#   ./deploy-adaptive-epp.sh <image-tag>
#   ./deploy-adaptive-epp.sh ttl.sh/llm-d-epp-adaptive-v2:24h
#
# To revert to static EPP:
#   ./deploy-adaptive-epp.sh --revert

NAMESPACE="${NAMESPACE:-kpouget-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFESTS="$SCRIPT_DIR/../manifests"

IMAGE="${1:?Usage: $0 <image-tag> or $0 --revert}"

if [ "$IMAGE" = "--revert" ]; then
  echo "=== Reverting to static EPP (ghcr.io/llm-d/llm-d-inference-scheduler:v0.7.0) ==="
  kubectl set image deployment/gaie-pd-epp -n "$NAMESPACE" \
    epp=ghcr.io/llm-d/llm-d-inference-scheduler:v0.7.0

  # Remove command override (static image has its own entrypoint)
  kubectl patch deployment gaie-pd-epp -n "$NAMESPACE" --type json \
    -p '[{"op":"remove","path":"/spec/template/spec/containers/0/command"}]' 2>/dev/null || true

  # Switch config back to pd-config.yaml
  kubectl patch deployment gaie-pd-epp -n "$NAMESPACE" --type json \
    -p "[$(kubectl get deployment gaie-pd-epp -n "$NAMESPACE" -o json | \
      python3 -c "
import json,sys
d=json.load(sys.stdin)
args=d['spec']['template']['spec']['containers'][0]['args']
for i,a in enumerate(args):
  if a.startswith('/config/adaptive'):
    print(json.dumps({'op':'replace','path':f'/spec/template/spec/containers/0/args/{i}','value':'/config/pd-config.yaml'}))
    break
")]" 2>/dev/null || true

  echo "Reverted. Waiting for rollout..."
  kubectl rollout status deployment/gaie-pd-epp -n "$NAMESPACE" --timeout=60s
  exit 0
fi

echo "=== Deploying adaptive EPP scorer ==="
echo "  Image: $IMAGE"

# Step 1: Add adaptive config to the EPP ConfigMap
echo "[1/3] Updating ConfigMap with adaptive-pd-config.yaml..."
ADAPTIVE_CONFIG=$(cat "$MANIFESTS/epp-adaptive-config.yaml" | python3 -c "
import sys, yaml, json
# Read the YAML config and format as a single-line YAML string for the ConfigMap patch
content = sys.stdin.read()
# Escape for JSON embedding
print(json.dumps(content))
" 2>/dev/null || cat "$MANIFESTS/epp-adaptive-config.yaml" | sed 's/$/\\n/' | tr -d '\n')

# Use a pre-built config inline
kubectl patch configmap gaie-pd-epp -n "$NAMESPACE" --type merge -p "
{
  \"data\": {
    \"adaptive-pd-config.yaml\": \"apiVersion: inference.networking.x-k8s.io/v1alpha1\nkind: EndpointPickerConfig\nfeatureGates:\n- prepareDataPlugins\nplugins:\n- type: prefill-header-handler\n- type: adaptive-scorer\n  parameters:\n    alpha: 0.15\n    queueVarianceThreshold: 4.0\n    kvUtilSpreadThreshold: 0.3\n    cacheHitRateThreshold: 0.2\n    queueDeferThreshold: 4.0\n    deferDuration: \\\"10ms\\\"\n    prefixBaseWeight: 3.0\n    loadBaseWeight: 2.0\n    activeBaseWeight: 2.0\n    enableWorkloadProfiles: true\n- type: prefix-cache-scorer\n  parameters:\n    maxPrefixBlocksToMatch: 256\n    lruCapacityPerServer: 31250\n- type: queue-scorer\n- type: prefill-filter\n- type: decode-filter\n- type: max-score-picker\n- type: prefix-based-pd-decider\n  parameters:\n    nonCachedTokens: 16\n- type: pd-profile-handler\n  parameters:\n    primaryPort: 0\n    deciderPluginName: prefix-based-pd-decider\nschedulingProfiles:\n- name: prefill\n  plugins:\n  - pluginRef: prefill-filter\n  - pluginRef: max-score-picker\n  - pluginRef: adaptive-scorer\n    weight: 3\n  - pluginRef: queue-scorer\n    weight: 1\n- name: decode\n  plugins:\n  - pluginRef: decode-filter\n  - pluginRef: max-score-picker\n  - pluginRef: adaptive-scorer\n    weight: 3\n  - pluginRef: queue-scorer\n    weight: 1\n\"
  }
}"

# Step 2: Swap image and set entrypoint
echo "[2/3] Swapping EPP image..."
kubectl set image deployment/gaie-pd-epp -n "$NAMESPACE" epp="$IMAGE"

# Ensure command is set (crane-built images have no entrypoint)
kubectl patch deployment gaie-pd-epp -n "$NAMESPACE" --type json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/command","value":["/app/epp"]}]' 2>/dev/null || \
kubectl patch deployment gaie-pd-epp -n "$NAMESPACE" --type json \
  -p '[{"op":"replace","path":"/spec/template/spec/containers/0/command","value":["/app/epp"]}]'

# Step 3: Switch config file arg
echo "[3/3] Switching to adaptive config..."
# Find and replace the --config-file arg value
ARGS_JSON=$(kubectl get deployment gaie-pd-epp -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].args}')
CONFIG_IDX=$(echo "$ARGS_JSON" | python3 -c "
import json, sys
args = json.loads(sys.stdin.read())
for i, a in enumerate(args):
    if a.startswith('/config/'):
        print(i)
        break
")
if [ -n "$CONFIG_IDX" ]; then
  kubectl patch deployment gaie-pd-epp -n "$NAMESPACE" --type json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args/$CONFIG_IDX\",\"value\":\"/config/adaptive-pd-config.yaml\"}]"
fi

echo ""
echo "Waiting for rollout..."
kubectl rollout status deployment/gaie-pd-epp -n "$NAMESPACE" --timeout=120s

echo ""
echo "=== Adaptive EPP deployed ==="
echo "Verify:  kubectl logs -n $NAMESPACE deployment/gaie-pd-epp --tail=5"
echo "Test:    kubectl port-forward -n $NAMESPACE svc/infra-gptoss-inference-gateway-istio 8080:80"
echo "Revert:  $0 --revert"
