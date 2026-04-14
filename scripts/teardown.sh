#!/usr/bin/env bash
set -euo pipefail

# Tear down the entire llm-d stack.

NAMESPACE="${NAMESPACE:-kpouget-dev}"

echo "=== Tearing down llm-d stack in $NAMESPACE ==="

for release in ms-pd ms-gptoss gaie-pd gaie-gptoss infra-gptoss; do
  if helm status "$release" -n "$NAMESPACE" &>/dev/null; then
    echo "Uninstalling $release..."
    helm uninstall "$release" -n "$NAMESPACE"
  fi
done

kubectl delete httproute gptoss-route -n "$NAMESPACE" 2>/dev/null || true

echo "Done. Verify with: kubectl get all -n $NAMESPACE"
