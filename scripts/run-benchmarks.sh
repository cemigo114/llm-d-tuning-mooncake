#!/usr/bin/env bash
set -euo pipefail

# Run Mooncake trace replay benchmarks against the llm-d stack.
#
# Prerequisites:
#   - pip install aiohttp
#   - Port-forward running: kubectl port-forward svc/infra-gptoss-inference-gateway-istio 8080:80
#   - Mooncake repo cloned (for trace files)
#
# Usage:
#   ./run-benchmarks.sh                          # run all
#   ./run-benchmarks.sh conversation             # conversation trace only
#   ./run-benchmarks.sh toolagent                # toolagent trace only
#   ENDPOINT=http://10.0.1.100:80 ./run-benchmarks.sh  # custom endpoint

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPLAY="$SCRIPT_DIR/../benchmarks/scripts/mooncake_replay.py"
RESULTS_DIR="$SCRIPT_DIR/../benchmarks/results"
TRACES_DIR="${MOONCAKE_TRACES:-$SCRIPT_DIR/../../Mooncake/FAST25-release/traces}"

ENDPOINT="${ENDPOINT:-http://localhost:8080}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
MAX_REQUESTS="${MAX_REQUESTS:-500}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
RATE_SCALE="${RATE_SCALE:-0.0005}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULTS_DIR"

run_trace() {
  local trace_name="$1"
  local trace_file="$TRACES_DIR/${trace_name}_trace.jsonl"
  local output="$RESULTS_DIR/${trace_name}_${TAG}.json"

  if [ ! -f "$trace_file" ]; then
    echo "ERROR: Trace file not found: $trace_file"
    echo "Set MOONCAKE_TRACES to the directory containing the trace files."
    return 1
  fi

  echo ""
  echo "=========================================="
  echo "  Running: $trace_name trace"
  echo "  Endpoint: $ENDPOINT"
  echo "  Model: $MODEL"
  echo "  Requests: $MAX_REQUESTS"
  echo "  Concurrency: $MAX_CONCURRENCY"
  echo "  Output: $output"
  echo "=========================================="

  python3 "$REPLAY" \
    --trace "$trace_file" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --rate-scale "$RATE_SCALE" \
    --max-requests "$MAX_REQUESTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --output "$output"
}

# Quick connectivity check
echo "Checking endpoint connectivity..."
if ! curl -s --max-time 5 "$ENDPOINT/v1/models" > /dev/null 2>&1; then
  echo "ERROR: Cannot reach $ENDPOINT/v1/models"
  echo "Start port-forward: kubectl port-forward -n \$NAMESPACE svc/infra-gptoss-inference-gateway-istio 8080:80"
  exit 1
fi
echo "Endpoint OK: $(curl -s "$ENDPOINT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo 'connected')"

TRACE_FILTER="${1:-all}"

case "$TRACE_FILTER" in
  conversation)
    run_trace "conversation"
    ;;
  toolagent)
    run_trace "toolagent"
    ;;
  synthetic)
    run_trace "synthetic"
    ;;
  all)
    run_trace "conversation"
    run_trace "toolagent"
    ;;
  *)
    echo "Usage: $0 [conversation|toolagent|synthetic|all]"
    exit 1
    ;;
esac

echo ""
echo "=== Benchmarks complete ==="
echo "Results in: $RESULTS_DIR/"
echo "Analyze with: python3 $SCRIPT_DIR/../benchmarks/scripts/analyze_results.py $RESULTS_DIR/ --format markdown"
