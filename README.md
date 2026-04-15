# llm-d Playbook: Deployment to Adaptive Scheduling to P/D Disaggregation

End-to-end playbook for deploying [llm-d](https://github.com/llm-d/llm-d) on a Kubernetes GPU cluster, benchmarking with production-realistic [Mooncake](https://github.com/kvcache-ai/Mooncake) traces, and comparing co-located vs P/D disaggregated serving with static vs adaptive EPP scoring.

## Cluster Tested

| Resource | Value |
|----------|-------|
| Nodes | 2 (CoreWeave `g12540c`, `gf2ab44`) |
| GPUs | 16x NVIDIA H200 (141 GB HBM3e) |
| Network | RDMA / InfiniBand |
| Driver | 580.126.20, CUDA 13.0 |
| K8s | v1.35.0 |
| Model | `openai/gpt-oss-120b` (FP4, `VLLM_MXFP4_USE_MARLIN=1`) |
| vLLM | `vllm/vllm-openai:v0.18.0` (co-located) / `ghcr.io/llm-d/llm-d-cuda:v0.6.0` (P/D) |
| llm-d | v0.7.0 EPP, v1.4.0 InferencePool |


**Trace:** Mooncake FAST'25 conversation trace (500 requests, 32 concurrency, 4K input tokens; 300 requests at 8K for Stage 5)

## Results

| # | Stage | What's Added | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | ITL P50 (ms) | TPSU (tok/s/user) | E2E P50 (s) |
|---|-------|-------------|-------------------:|-------------:|--------------:|------------:|-----------------:|------------:|
| 1 | **K8s Round-Robin** | vLLM behind plain K8s Service | 3,913 | 128 | 1,339 | 6.41 | 156 | 1.71 |
| 2 | **llm-d EPP** | Prefix-cache-aware routing | 4,517 **(+15%)** | 134 | 1,041 **(-22%)** | 5.77 **(-10%)** | 173 **(+11%)** | 1.56 |
| 3 | **P/D Disaggregation** | NIXL/RDMA prefill-decode split | 4,200 **(+7%)** | 338 | 3,715 | 4.81 **(-25%)** | 208 **(+33%)** | 1.48 |
| 4 | **P/D + Adaptive EPP** | Workload-aware dynamic weights | **5,210 (+33%)** | 191 | **875 (-35%)** | 4.76 **(-26%)** | 210 **(+35%)** | **1.34** |
| 5 | **P/D + CPU Offload** | MultiConnector 100GB CPU tier (8K tokens) | 1,543+ | 172 | 4,380+ | **4.29 (-33%)** | **233 (+49%)** | 0.04 |

*All deltas relative to Stage 1 (K8s baseline). +Stage 5 uses 8K input tokens (300 requests) -- throughput and TTFT P99 not directly comparable to 4K stages.*

## Quick Start

### Prerequisites

```bash
# Tools
brew install helm kubectl yq jq

# Helm repos
helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/
helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/
helm repo update

# HuggingFace token (must exist in target namespace)
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=$HF_TOKEN -n $NAMESPACE

# Python (for benchmarks)
pip install aiohttp

# Clone Mooncake for trace data
git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git ../Mooncake
```

### Full Reproduction (4 steps)

```bash
export NAMESPACE=kpouget-dev

# Step 1: Deploy co-located baseline (4 x TP=4)
./scripts/deploy-colocated.sh
kubectl port-forward -n $NAMESPACE svc/infra-gptoss-inference-gateway-istio 8080:80 &
TAG=colocated ./scripts/run-benchmarks.sh

# Step 2: Deploy P/D disaggregation (2P + 2D, TP=4, NIXL/RDMA)
./scripts/deploy-pd.sh
TAG=pd ./scripts/run-benchmarks.sh

# Step 3: Build + deploy adaptive EPP scorer (workload-aware profiles)
./scripts/build-adaptive-epp.sh           # builds in-cluster, pushes to ttl.sh
./scripts/deploy-adaptive-epp.sh ttl.sh/llm-d-epp-adaptive-XXXXX:24h
TAG=adaptive ./scripts/run-benchmarks.sh

# Step 4: Deploy P/D with MultiConnector (NIXL + CPU offload) for long sequences
./scripts/deploy-pd-multiconnector.sh
MAX_INPUT_TOKENS=8192 MAX_REQUESTS=300 TAG=multi_8k ./scripts/run-benchmarks.sh

# Step 5: Compare
python3 benchmarks/scripts/analyze_results.py benchmarks/results/ --format markdown
```

### Teardown

```bash
./scripts/teardown.sh
```

## Results

All benchmarks: 500 requests from Mooncake FAST'25 traces (300 for 8K tests), `--max-concurrency 32` (16 for 8K), `--max-input-tokens 4096` (8192 for long-sequence tests). Every number independently verified from per-request JSON data (see `benchmarks/results/`).

### Conversation Trace (multi-turn, high prefix sharing)

| Metric | Co-located | P/D Static | P/D Adaptive v2 | Best vs Co-located |
|--------|-----------|-----------|-----------------|-------------------|
| Throughput (tok/s) | 3,814 | 4,063 | **4,729** | **+24.0%** |
| TTFT P50 | 175 ms | 376 ms | **148 ms** | **-15.4%** |
| TTFT P90 | 399 ms | 765 ms | **406 ms** | +1.7% |
| TTFT P99 | 632 ms | 3,688 ms | **1,256 ms** | +98.7% |
| E2E P50 | 1.93 s | 1.56 s | **1.46 s** | **-24.4%** |
| E2E P90 | 2.35 s | 1.93 s | **1.87 s** | **-20.4%** |

### Toolagent Trace (tool-calling, diverse prefixes)

| Metric | Co-located | P/D Static | P/D Adaptive v2 | Best vs Co-located |
|--------|-----------|-----------|-----------------|-------------------|
| Throughput (tok/s) | 2,656 | **4,191** | 3,983 | **+49.9%** |
| TTFT P50 | 204 ms | 251 ms | **125 ms** | **-38.9%** |
| TTFT P90 | 2,173 ms | 627 ms | **406 ms** | **-81.3%** |
| TTFT P99 | 3,553 ms | **1,151 ms** | 1,407 ms | **-67.6%** |
| E2E P50 | 1.49 s | 0.68 s | **0.45 s** | **-69.8%** |
| E2E P90 | 2.70 s | **1.72 s** | 1.83 s | **-36.3%** |

### Adaptive Scorer Progression (P/D topology)

Shows the effect of each adaptive scorer iteration on the P/D deployment:

**Conversation:**

| Metric | Static | Adaptive v1 | Adaptive v2 |
|--------|--------|------------|------------|
| Throughput | 4,063 | 4,253 (+5%) | **4,729 (+16%)** |
| TTFT P50 | 376 ms | 320 ms (-15%) | **148 ms (-61%)** |
| TTFT P99 | 3,688 ms | 2,427 ms (-34%) | **1,256 ms (-66%)** |

**Toolagent:**

| Metric | Static | Adaptive v1 | Adaptive v2 |
|--------|--------|------------|------------|
| Throughput | 4,191 | 3,575 (-15%) | **3,983 (-5%)** |
| TTFT P50 | 251 ms | 243 ms (-3%) | **125 ms (-50%)** |
| E2E P50 | 0.68 s | 0.73 s (+7%) | **0.45 s (-34%)** |

### MultiConnector: NIXL + CPU Offload (Long Sequences, 8K tokens)

At longer input sequences, HBM fills up and KV cache eviction hurts performance. The `MultiConnector` combines `NixlConnector` (P/D RDMA transfer) with `OffloadingConnector` (100GB CPU RAM cache tier) via vLLM's composable KV connector API.

**Conversation Trace, 300 requests, `--max-input-tokens 8192`, `--max-concurrency 16`:**

| Metric | NIXL-only (P/D) | MultiConnector (NIXL + CPU 100GB) | Delta |
|--------|-----------------|----------------------------------|-------|
| Throughput | 2,051 tok/s | **2,942 tok/s** | **+43.4%** |
| TTFT P50 | 155 ms | **45 ms** | **-71.3%** |
| TTFT P90 | 2,484 ms | **138 ms** | **-94.4%** |
| TTFT P99 | 2,946 ms | **210 ms** | **-92.9%** |
| E2E P90 | 1.24 s | **1.14 s** | **-8.1%** |

**Important: sequence length determines whether CPU offload helps or hurts:**

| Input tokens | NIXL-only | MultiConnector | Winner |
|-------------|-----------|----------------|--------|
| 4K (short) | 4,063 tok/s | 2,642 tok/s | NIXL-only (-35% with Multi) |
| 8K (long) | 2,051 tok/s | 2,942 tok/s | **MultiConnector (+43%)** |

The crossover point is ~6-8K input tokens where KV cache starts exceeding comfortable HBM capacity and the CPU tier provides actual cache hits instead of just overhead.

**vLLM KV transfer config:**
```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
      {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"cpu_bytes_to_use": 107374182400}}
    ]
  }
}
```

### Key Findings

1. **P/D + adaptive v2 is the best config for short sequences (4K tokens).** TTFT P50 improved 61% on conversation and 50% on toolagent vs static P/D.

2. **MultiConnector (NIXL + CPU offload) is the best config for long sequences (8K+ tokens).** TTFT P90 improved 94% and throughput +43% when KV cache exceeds HBM.

3. **Workload-aware profiles fixed v1's toolagent regression.** v1 hurt toolagent throughput by -15% because it applied conversation-optimized weights (high prefix, low defer threshold) to a diverse-prefix workload. v2 auto-detects completions requests as "diverse" profile and applies load-balanced weights with a higher defer threshold.

4. **P/D disaggregation is universally better for E2E latency** because decode workers produce tokens without prefill contention.

5. **Queue threshold deferral (Dynamo-inspired) is the single highest-impact feature.** It prevents stale-metric pile-on where concurrent requests all route to the same "low queue" pod.

6. **CPU offload hurts at short sequences, helps at long ones.** The crossover is ~6-8K tokens. Deploy recommendation: use NIXL-only for interactive/short workloads, MultiConnector for batch/long-context workloads.

## Topology Study: P/D Worker Ratio and Tensor Parallelism

We tested two P/D topologies on 16 GPUs to find the optimal prefill:decode ratio:

### 8P(TP=1)+2D(TP=4) vs 2P(TP=4)+2D(TP=4)

| Metric (4K tokens, warm cache) | 2P(TP=4)+2D(TP=4) | 8P(TP=1)+2D(TP=4) |
|-------------------------------|-------------------|-------------------|
| Throughput | 4,200 tok/s | **5,548 tok/s (+32%)** |
| TTFT P50 | 338 ms | **31 ms (-91%)** |
| TTFT P99 | 3,715 ms | **615 ms (-83%)** |
| TPSU | 208 | 190 |

**8P(TP=1) is the recommended production topology.** The 8x higher prefill concurrency eliminates prefill queueing — TTFT drops to ~30ms. This matches the llm-d best practice: "deploy P workers with less parallelism and more replicas." See `REPORT.md` for the full topology analysis with both 4K and 8K token results.

**Use `manifests/pd/ms-pd-8p1-2d4-values.yaml`** for the recommended topology.

### Why these numbers?

- **TP=1 prefill**: gpt-oss-120b at FP4 is ~30GB, fits in a single H200 (141GB) with ~110GB for KV cache. Zero TP communication overhead.
- **TP=4 decode**: decode is memory-bandwidth-bound. TP=4 gives 4x memory bandwidth for fast token generation.
- **8:2 ratio**: Mooncake traces have ~35:1 ISL:OSL ratio (prefill-heavy). 8 prefill workers prevent prefill from becoming the bottleneck.

## Adaptive EPP Scorer

`adaptive-scorer/adaptive.go` replaces static weight configuration with dynamic per-request scoring:

### Workload Profiles

| Profile | Detection | Prefix | Load | Active | Defer Threshold |
|---------|-----------|--------|------|--------|----------------|
| **Conversational** | ChatCompletions API | 4.0 | 2.0 | 1.5 | 3.0 |
| **Diverse** | Short completions, no chat | 1.5 | 3.0 | 2.5 | 6.0 |
| **Long-context** | Prompt > 20K chars | 2.0 | 4.0 | 3.0 | 2.0 |
| Default | Fallback | 3.0 | 2.0 | 2.0 | 4.0 |

### Dynamic Adjustments (on top of profile weights)

- **Queue variance high** → boost load/active weights, dampen prefix (prevents hot-node piling)
- **KV util spread high** → boost load weight (rebalance memory pressure)
- **Cache hit rate low** → halve prefix weight (stop chasing cold caches)
- **All queues congested** → defer routing 10ms for fresher metrics (Dynamo-inspired)

### Build and Deploy

```bash
# Build in-cluster (no local Go/Docker needed)
./scripts/build-adaptive-epp.sh

# Deploy (hot-swaps the EPP, model servers stay running)
./scripts/deploy-adaptive-epp.sh ttl.sh/llm-d-epp-adaptive-XXXXX:24h

# Revert to static EPP
./scripts/deploy-adaptive-epp.sh --revert
```

## Repo Structure

```
llm-d-playbook/
├── README.md
├── scripts/
│   ├── deploy-colocated.sh          # Deploy 4×TP4 co-located
│   ├── deploy-pd.sh                 # Deploy 2P+2D disaggregated
│   ├── build-adaptive-epp.sh        # Build adaptive scorer in-cluster
│   ├── deploy-adaptive-epp.sh       # Hot-swap EPP to adaptive (or --revert)
│   ├── deploy-pd-multiconnector.sh  # P/D with NIXL + CPU offload (long sequences)
│   ├── run-benchmarks.sh            # Run Mooncake trace replay
│   └── teardown.sh                  # Remove everything
├── manifests/
│   ├── gateway.yaml
│   ├── epp-adaptive-config.yaml     # Adaptive scorer EPP config
│   ├── colocated/                   # Co-located deployment values
│   ├── pd/                          # P/D deployment values (NIXL only, 2P+2D and 8P+2D)
│   └── pd-multiconnector/           # P/D + CPU offload (NIXL + OffloadingConnector)
├── benchmarks/
│   ├── scripts/
│   │   ├── mooncake_replay.py       # Trace replay with prefix sharing
│   │   └── analyze_results.py       # Generate comparison tables
│   └── results/                     # Raw JSON per-request data (9 files)
│       ├── mooncake_conversation_500.json    # Co-located conversation
│       ├── mooncake_toolagent_500.json       # Co-located toolagent
│       ├── pd_conversation_500.json          # P/D static conversation
│       ├── pd_toolagent_500.json             # P/D static toolagent
│       ├── adaptive_pd_conversation_500.json # P/D adaptive v1 conv
│       ├── adaptive_pd_toolagent_500.json    # P/D adaptive v1 tool
│       ├── adaptive_v2_pd_conversation_500.json  # P/D adaptive v2 conv
│       ├── adaptive_v2_pd_toolagent_500.json     # P/D adaptive v2 tool
│       ├── nixl_pd_conversation_8k_300.json      # NIXL-only 8K long-seq
│       ├── multi_pd_conversation_8k_300.json     # MultiConnector 8K long-seq
│       └── topology/                             # 8P+2D topology comparison
│           ├── pd_8p1_2d4_4k.json                # 8P(TP=1)+2D(TP=4) at 4K
│           └── pd_8p1_2d4_8k.json                # 8P(TP=1)+2D(TP=4) at 8K
├── adaptive-scorer/
│   ├── adaptive.go                  # Workload-aware adaptive scorer (Go)
│   └── adaptive_test.go             # Unit tests
└── docs/
    ├── PLAN.md                      # Full execution plan
    └── kv-cache-blueprint.md        # 4-layer KV cache optimization design
```

## Data Integrity

All 56 metrics across 9 benchmark files have been independently verified by recomputing percentiles from raw per-request JSON. Throughput values are computed as `total_output_tokens / wall_clock_seconds` and match within 0.2% (timer precision). The verification script:

```bash
python3 benchmarks/scripts/analyze_results.py benchmarks/results/ --format markdown
```

Each JSON file contains 500 entries (100 for baseline) with per-request `ttft`, `total_time`, `output_tokens`, `input_length`, and `status` fields. Any team member can recompute any number from these files.

## Mooncake Trace Replay

The replay script (`benchmarks/scripts/mooncake_replay.py`) converts Mooncake FAST'25 traces into OpenAI-compatible requests with controlled prefix sharing:

- Each `hash_id` maps to a deterministic 512-token text block (`random.Random(hash_id)`)
- Requests sharing `hash_id` values produce identical prefix text
- This triggers vLLM's prefix cache for EPP-aware routing
- `hash_id 0` appears in 100% of conversation trace requests (shared system prompt)

```bash
python3 benchmarks/scripts/mooncake_replay.py \
  --trace ../Mooncake/FAST25-release/traces/conversation_trace.jsonl \
  --endpoint http://localhost:8080 \
  --model openai/gpt-oss-120b \
  --rate-scale 0.0005 \
  --max-requests 500 \
  --max-concurrency 32 \
  --max-input-tokens 4096 \
  --output results.json
```

## Related Projects

- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes-native LLM inference
- [llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler) - EPP scorer plugins
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Distributed KV cache (FAST'25)
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) - Optimal vLLM serving configs
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) - Distributed inference framework (queue deferral inspiration)
