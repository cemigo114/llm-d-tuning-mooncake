#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# llm-d Playbook: End-to-End 5-Stage Benchmark
#
# Runs all 5 optimization stages sequentially on the same cluster, collecting
# fresh benchmark data at each stage. Produces a final comparison report.
#
# Stages:
#   1. K8s round-robin baseline (vLLM direct, no EPP)
#   2. llm-d co-located with EPP (prefix-cache-aware routing)
#   3. P/D disaggregation (NIXL/RDMA)
#   4. P/D + adaptive EPP (workload-aware profiles)
#   5. P/D + MultiConnector CPU offload (long sequences)
#
# Prerequisites:
#   - KUBECONFIG pointing to cluster with 16 GPUs and RDMA
#   - NAMESPACE set (default: kpouget-dev)
#   - llm-d-hf-token secret in namespace
#   - Helm repos: llm-d-infra, llm-d-modelservice
#   - Infrastructure chart already deployed (deploy-colocated.sh does this)
#   - Adaptive EPP image built (build-adaptive-epp.sh) — set ADAPTIVE_IMAGE
#
# Usage:
#   export NAMESPACE=kpouget-dev
#   export ADAPTIVE_IMAGE=ttl.sh/llm-d-epp-adaptive-v2:24h
#   ./scripts/run-all-stages.sh
##############################################################################

NAMESPACE="${NAMESPACE:-kpouget-dev}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFESTS="$SCRIPT_DIR/../manifests"
BENCH_SCRIPTS="$SCRIPT_DIR/../benchmarks/scripts"
RESULTS_DIR="$SCRIPT_DIR/../benchmarks/results/stages"
TRACE_DIR="${MOONCAKE_TRACES:-$SCRIPT_DIR/../../Mooncake/FAST25-release/traces}"
ADAPTIVE_IMAGE="${ADAPTIVE_IMAGE:-}"

MODEL="openai/gpt-oss-120b"
TRACE="$TRACE_DIR/conversation_trace.jsonl"
REQUESTS=500
CONCURRENCY=32
INPUT_TOKENS=4096
RATE_SCALE=0.0005

# Stage 5 uses longer sequences
STAGE5_INPUT_TOKENS=8192
STAGE5_REQUESTS=300
STAGE5_CONCURRENCY=16

mkdir -p "$RESULTS_DIR"

KC="kubectl -n $NAMESPACE"

##############################################################################
# Helper functions
##############################################################################

wait_for_pods() {
  local label="$1"
  local timeout="${2:-600}"
  echo "  Waiting for pods ($label)..."
  $KC wait --for=condition=Ready pod -l "$label" --timeout="${timeout}s"
}

wait_for_model_ready() {
  local svc="$1"
  local max_wait=600
  local waited=0
  echo "  Waiting for model to be ready on $svc..."
  while [ $waited -lt $max_wait ]; do
    if $KC exec bench-runner -- python3 -c "
import urllib.request
try:
    urllib.request.urlopen('http://${svc}/v1/models', timeout=5)
    exit(0)
except:
    exit(1)
" 2>/dev/null; then
      echo "  Model ready."
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    echo "    ... $waited/${max_wait}s"
  done
  echo "  ERROR: Model not ready after ${max_wait}s"
  return 1
}

run_benchmark() {
  local stage="$1"
  local endpoint="$2"
  local output="$3"
  local input_tokens="${4:-$INPUT_TOKENS}"
  local requests="${5:-$REQUESTS}"
  local concurrency="${6:-$CONCURRENCY}"

  echo ""
  echo "  Running benchmark: $stage"
  echo "    Endpoint: $endpoint"
  echo "    Requests: $requests, Concurrency: $concurrency, Max input: $input_tokens"

  $KC exec bench-runner -- python3 /bench/benchmark_stage.py \
    --trace /bench/conversation_trace.jsonl \
    --endpoint "$endpoint" \
    --model "$MODEL" \
    --stage "$stage" \
    --rate-scale "$RATE_SCALE" \
    --max-requests "$requests" \
    --max-concurrency "$concurrency" \
    --max-input-tokens "$input_tokens" \
    --output "/bench/results/$(basename "$output")"

  # Copy result back
  $KC cp "bench-runner:/bench/results/$(basename "$output")" "$output"
  echo "  Result saved to: $output"
}

teardown_model() {
  echo "  Tearing down model server..."
  for release in ms-gptoss ms-pd; do
    helm uninstall "$release" -n "$NAMESPACE" 2>/dev/null || true
  done
  for release in gaie-gptoss gaie-pd; do
    helm uninstall "$release" -n "$NAMESPACE" 2>/dev/null || true
  done
  $KC delete httproute gptoss-route 2>/dev/null || true
  # Wait for pods to terminate
  sleep 15
  # Clean up stale ReplicaSets
  $KC delete pods -l 'llm-d.ai/inference-serving=true' --grace-period=5 2>/dev/null || true
  sleep 5
}

##############################################################################
# Setup: create in-cluster benchmark runner
##############################################################################

echo "========================================================================"
echo "  SETUP: Creating in-cluster benchmark runner"
echo "========================================================================"

$KC apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: bench-runner
spec:
  restartPolicy: Never
  containers:
  - name: bench
    image: python:3.12-slim
    command: ["sleep", "86400"]
    volumeMounts:
    - name: data
      mountPath: /bench
  volumes:
  - name: data
    emptyDir:
      sizeLimit: 2Gi
EOF

$KC wait --for=condition=Ready pod/bench-runner --timeout=120s
$KC exec bench-runner -- pip install -q aiohttp 2>&1 | tail -1
$KC exec bench-runner -- mkdir -p /bench/results

# Copy benchmark script and trace data
$KC cp "$BENCH_SCRIPTS/benchmark_stage.py" bench-runner:/bench/benchmark_stage.py
$KC cp "$TRACE" bench-runner:/bench/conversation_trace.jsonl

echo "  Benchmark runner ready."

##############################################################################
# STAGE 1: Kubernetes round-robin baseline (no EPP)
##############################################################################

echo ""
echo "========================================================================"
echo "  STAGE 1: K8s Round-Robin Baseline (vLLM direct, no EPP)"
echo "========================================================================"

teardown_model

# Deploy vLLM model server only (no EPP, no InferencePool)
echo "  Deploying vLLM model server (4×TP4, no EPP)..."
helm install ms-gptoss llm-d-modelservice/llm-d-modelservice \
  -n "$NAMESPACE" -f "$MANIFESTS/colocated/ms-values.yaml"

# Create a plain K8s Service for round-robin (bypasses EPP entirely)
$KC apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-direct
spec:
  selector:
    llm-d.ai/inference-serving: "true"
    llm-d.ai/guide: "gptoss-cks"
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
EOF

wait_for_pods "llm-d.ai/guide=gptoss-cks"
wait_for_model_ready "vllm-direct:8000"

run_benchmark \
  "1-k8s-roundrobin" \
  "http://vllm-direct.${NAMESPACE}.svc.cluster.local:8000" \
  "$RESULTS_DIR/stage_1_k8s_roundrobin.json"

##############################################################################
# STAGE 2: llm-d co-located with EPP
##############################################################################

echo ""
echo "========================================================================"
echo "  STAGE 2: llm-d Co-located with EPP (prefix-cache-aware routing)"
echo "========================================================================"

# Add EPP on top of existing model server (don't redeploy vLLM)
echo "  Deploying EPP inference scheduler..."
helm install gaie-gptoss \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --version v1.4.0 -n "$NAMESPACE" \
  -f "$MANIFESTS/colocated/gaie-values.yaml" \
  --set 'provider.name=istio' \
  --set "provider.istio.destinationRule.host=gaie-gptoss-epp.${NAMESPACE}.svc.cluster.local" \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.http2MaxRequests=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.tcp.maxConnections=256000'

$KC apply -f "$MANIFESTS/colocated/httproute.yaml"
sleep 10

GATEWAY_SVC="infra-gptoss-inference-gateway-istio.${NAMESPACE}.svc.cluster.local:80"
wait_for_model_ready "$GATEWAY_SVC"

run_benchmark \
  "2-llmd-colocated-epp" \
  "http://$GATEWAY_SVC" \
  "$RESULTS_DIR/stage_2_llmd_colocated.json"

##############################################################################
# STAGE 3: P/D Disaggregation
##############################################################################

echo ""
echo "========================================================================"
echo "  STAGE 3: P/D Disaggregation (2P+2D, NIXL/RDMA)"
echo "========================================================================"

teardown_model

# Deploy P/D EPP
echo "  Deploying P/D EPP..."
helm install gaie-pd \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  --version v1.4.0 -n "$NAMESPACE" \
  -f "$MANIFESTS/pd/gaie-pd-values.yaml" \
  --set 'provider.name=istio' \
  --set "provider.istio.destinationRule.host=gaie-pd-epp.${NAMESPACE}.svc.cluster.local" \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.http.http2MaxRequests=256000' \
  --set 'provider.istio.destinationRule.trafficPolicy.connectionPool.tcp.maxConnections=256000'

# Deploy P/D model server (NIXL only)
echo "  Deploying P/D model server (2P+2D, TP4, NIXL)..."
helm install ms-pd llm-d-modelservice/llm-d-modelservice \
  -n "$NAMESPACE" -f "$MANIFESTS/pd/ms-pd-values.yaml"

# Update HTTPRoute to P/D pool
$KC apply -f "$MANIFESTS/pd/httproute-pd.yaml"

wait_for_pods "llm-d.ai/guide=pd-gptoss-cks" 600
wait_for_model_ready "$GATEWAY_SVC"

run_benchmark \
  "3-pd-disaggregation" \
  "http://$GATEWAY_SVC" \
  "$RESULTS_DIR/stage_3_pd_disagg.json"

##############################################################################
# STAGE 4: P/D + Adaptive EPP
##############################################################################

echo ""
echo "========================================================================"
echo "  STAGE 4: P/D + Adaptive EPP (workload-aware profiles)"
echo "========================================================================"

if [ -z "$ADAPTIVE_IMAGE" ]; then
  echo "  ADAPTIVE_IMAGE not set. Building..."
  # Build in-cluster
  "$SCRIPT_DIR/build-adaptive-epp.sh"
  ADAPTIVE_IMAGE=$(grep "^IMAGE=" /tmp/adaptive-build-output 2>/dev/null | cut -d= -f2 || echo "")
  if [ -z "$ADAPTIVE_IMAGE" ]; then
    echo "  ERROR: Failed to build adaptive EPP. Skipping stage 4."
    echo '{"stage":"4-pd-adaptive-epp","error":"build failed"}' > "$RESULTS_DIR/stage_4_pd_adaptive.json"
    # Continue to stage 5
  fi
fi

if [ -n "$ADAPTIVE_IMAGE" ]; then
  echo "  Deploying adaptive EPP: $ADAPTIVE_IMAGE"
  "$SCRIPT_DIR/deploy-adaptive-epp.sh" "$ADAPTIVE_IMAGE"
  sleep 10
  wait_for_model_ready "$GATEWAY_SVC"

  run_benchmark \
    "4-pd-adaptive-epp" \
    "http://$GATEWAY_SVC" \
    "$RESULTS_DIR/stage_4_pd_adaptive.json"

  # Revert to static EPP for stage 5
  echo "  Reverting to static EPP..."
  "$SCRIPT_DIR/deploy-adaptive-epp.sh" --revert
  sleep 10
fi

##############################################################################
# STAGE 5: P/D + MultiConnector CPU Offload (long sequences)
##############################################################################

echo ""
echo "========================================================================"
echo "  STAGE 5: P/D + MultiConnector CPU Offload (8K tokens)"
echo "========================================================================"

# Upgrade model server to MultiConnector
echo "  Upgrading to MultiConnector (NIXL + CPU 100GB)..."
helm upgrade ms-pd llm-d-modelservice/llm-d-modelservice \
  -n "$NAMESPACE" -f "$MANIFESTS/pd-multiconnector/ms-pd-multi-values.yaml"

# Force pod recreation (rolling update can't work — GPU contention)
$KC delete pods -l 'llm-d.ai/guide=pd-gptoss-cks' --grace-period=10 2>/dev/null || true
# Scale down old ReplicaSets
for rs in $($KC get rs -l 'llm-d.ai/guide=pd-gptoss-cks' -o name 2>/dev/null); do
  $KC scale "$rs" --replicas=0 2>/dev/null || true
done
sleep 15
# The new ReplicaSets will create pods
$KC get rs -l 'llm-d.ai/guide=pd-gptoss-cks' --sort-by=.metadata.creationTimestamp -o name | tail -2 | while read rs; do
  $KC scale "$rs" --replicas=2 2>/dev/null || true
done

wait_for_pods "llm-d.ai/guide=pd-gptoss-cks" 600
wait_for_model_ready "$GATEWAY_SVC"

run_benchmark \
  "5-pd-multiconnector-cpu-offload" \
  "http://$GATEWAY_SVC" \
  "$RESULTS_DIR/stage_5_pd_multiconnector.json" \
  "$STAGE5_INPUT_TOKENS" "$STAGE5_REQUESTS" "$STAGE5_CONCURRENCY"

##############################################################################
# REPORT
##############################################################################

echo ""
echo "========================================================================"
echo "  GENERATING REPORT"
echo "========================================================================"

# Copy all stage results from bench runner
for f in $($KC exec bench-runner -- ls /bench/results/ 2>/dev/null); do
  $KC cp "bench-runner:/bench/results/$f" "$RESULTS_DIR/$f" 2>/dev/null || true
done

# Generate report
python3 "$BENCH_SCRIPTS/generate_report.py" "$RESULTS_DIR"/stage_*.json

# Cleanup
echo ""
echo "  Cleaning up bench runner..."
$KC delete pod bench-runner 2>/dev/null || true
$KC delete svc vllm-direct 2>/dev/null || true

echo ""
echo "========================================================================"
echo "  ALL 5 STAGES COMPLETE"
echo "  Results: $RESULTS_DIR/"
echo "  Report:  python3 $BENCH_SCRIPTS/generate_report.py $RESULTS_DIR/stage_*.json"
echo "========================================================================"
