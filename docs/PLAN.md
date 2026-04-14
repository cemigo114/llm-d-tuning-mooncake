# llm-d Playbook: Deployment to Adaptive Scheduling to P/D Disaggregation

## Context

**Problem:** We have 1-2 day access to a CoreWeave cluster (2 nodes, 16x NVIDIA H200 141GB HBM3e, RDMA-capable) and need to validate the full llm-d inference stack — from deployment through intelligent EPP routing to prefill/decode disaggregation — using production-realistic Mooncake traces. Additionally, we need to improve llm-d's EPP scorer to be more adaptive (closing the gap with NVIDIA Dynamo's dynamic routing) and produce a KV cache optimization blueprint.

**Cluster:** `6787d4-b7d406b9.k8s.us-east-01a.coreweave.com`
- Nodes: `g12540c`, `gf2ab44` — 8x H200 each
- Driver: 580.126.20, CUDA 13.0, K8s v1.35.0
- Namespace: `kpouget-dev`
- GPUs currently held by `cw-hpc-verification` jobs (idle, will free)
- KUBECONFIG: `/Users/yfama/CKS/kubeconfig`

**Goal:** Produce benchmark data comparing llm-d configurations, a prototype adaptive EPP scorer, and a KV cache optimization blueprint — all within the cluster access window.

---

## Stage 0: Prerequisites & Cluster Preparation (Hour 0-1)

### 0.1 Wait for GPU availability
```bash
# Monitor HPC verification jobs
KUBECONFIG=/Users/yfama/CKS/kubeconfig kubectl get pods -n cw-hpc-verification -w
```
- If jobs don't complete soon, coordinate with cluster admin to preempt them
- Validate all 16 GPUs free with `nvidia-smi` via exec

### 0.2 Install dependencies
- [ ] Helm 3.x, `istioctl` or Envoy Gateway, `yq`, `jq`
- [ ] Clone repos:
  - `llm-d/llm-d` — main project
  - `llm-d-incubation/llm-d-infra` — Helm charts
  - `llm-d/llm-d-inference-scheduler` — EPP scorer source
  - `kvcache-ai/Mooncake` — trace data
  - `vllm-project/guidellm` — benchmarking
- [ ] HuggingFace token secret: `kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN -n kpouget-dev`

### 0.3 Select model
**Primary:** `gpt-oss-120b`
- 120B dense model, source from Red Hat model registry
- Use **InferenceX** (`SemiAnalysisAI/InferenceX`) for optimal vLLM serving config
- ~240GB at BF16 → TP=2 on H200 (282GB) tight, TP=4 (564GB) comfortable

**Topology options:**
| Config | Prefill | Decode | Replicas | Use Case |
|--------|---------|--------|----------|----------|
| Co-located | — | — | 2x (TP=4) per node | Baseline + EPP routing |
| P/D disagg | 8 GPUs node1 (TP=4, 2 replicas) | 8 GPUs node2 (TP=4, 2 replicas) | Split | P/D benchmarks |

---

## Stage 1: llm-d Deployment & Baseline (Hours 1-5)

### 1.1 Deploy Gateway API + llm-d control plane
- [ ] Install Gateway API CRDs (Istio or Envoy Gateway)
- [ ] Deploy llm-d inference scheduler via Helm:
  ```bash
  helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/
  helm install llm-d-scheduler llm-d-infra/llm-d-inference-scheduler \
    -n kpouget-dev --values values.yaml
  ```
- [ ] Deploy vLLM model servers (4 replicas, TP=2 each)
- [ ] Configure InferenceModel + InferencePool CRDs
- [ ] Validate: send test completion request through gateway

### 1.2 Baseline benchmark (random routing, no prefix caching)
- [ ] Run GuideLLM sweep with synthetic data:
  ```bash
  guidellm benchmark \
    --target "http://<gateway-endpoint>" \
    --data "prompt_tokens=4096,output_tokens=256" \
    --profile sweep
  ```
- [ ] Record: throughput (tok/s), TTFT P50/P90/P99, ITL, queue depths
- [ ] Save as `baseline_random.json`

### 1.3 Enable prefix caching
- [ ] Enable vLLM `--enable-prefix-caching` on all replicas
- [ ] Deploy KV event publisher sidecar
- [ ] Configure EPP with default weights: `prefix:3, kv-util:2, queue:2`
- [ ] Re-run GuideLLM sweep → save as `baseline_prefix_cache.json`

---

## Stage 2: Mooncake Trace Integration (Hours 5-8)

### 2.1 Prepare Mooncake trace adapter
GuideLLM does NOT natively support timestamp-based trace replay. We need a custom adapter.

**Trace format** (from `Mooncake/FAST25-release/traces/`):
```json
{"timestamp": 27482, "input_length": 6955, "output_length": 52, "hash_ids": [46, 47, 48, ...]}
```

**Adapter script** (`mooncake_replay.py`):
1. Parse `conversation_trace.jsonl` (multi-turn, best prefix-caching test)
2. Convert to GuideLLM dataset format (JSONL with `prompt` and `prompt_tokens` columns)
3. Use `hash_ids` to generate synthetic prompts with controlled prefix overlap (same hash_id blocks = shared prefix tokens)
4. Replay with Poisson arrival rate derived from trace timestamps
5. Alternatively: direct vLLM client replay script with precise timestamp control

**Key mapping:**
| Mooncake Field | GuideLLM/vLLM Mapping |
|---|---|
| `timestamp` | Inter-arrival time (delta between consecutive) |
| `input_length` | `max_tokens` for prompt generation / `prompt_tokens` |
| `output_length` | `max_tokens` response parameter |
| `hash_ids` | Synthetic prompt construction for prefix sharing |

### 2.2 Run Mooncake trace benchmarks
- [ ] Replay `conversation_trace.jsonl` against EPP with default weights
- [ ] Replay `toolagent_trace.jsonl` (different prefix pattern)
- [ ] Record cache hit rates, TTFT, throughput per trace type
- [ ] Save as `mooncake_conversation.json`, `mooncake_toolagent.json`

---

## Stage 3: Adaptive EPP Scorer (Hours 8-16) — CRITICAL PATH

### 3.1 Problem analysis

**Current llm-d EPP scoring** (static):
```
TotalScore = (PrefixCacheScore x W1) + (KVUtilScore x W2) + (QueueScore x W3)
Default: W1=3, W2=2, W3=2
```

**Dynamo's edge** — three key techniques llm-d lacks:
1. **Deferred routing with queue threshold** — when all workers exceed a queue threshold, delay routing to get fresher load metrics before dispatching
2. **Predictive load estimation** — Kalman filters to predict future load, not just react to current state
3. **Cost-function transparency** — single tunable `overlap_score_weight` with clear semantics

**GKE insight** — queue weight increase (3:3:2 → 3:5:2) prevented hot-node piling when prefix affinity dominated routing decisions.

### 3.2 Proposed adaptive scorer design

**File to modify:** `llm-d/llm-d-inference-scheduler` — EPP scorer pipeline

**Enhancement 1: Dynamic weight adjustment (EMA-based)**
```
# Observe system state every scoring cycle
queue_variance = var(queue_depths across pods)
cache_hit_rate = EMA(recent cache hits / total requests, alpha=0.1)
kv_util_spread = max(kv_util) - min(kv_util)

# Adjust weights dynamically
if queue_variance > HIGH_THRESHOLD:
    queue_weight = base_queue_weight * (1 + queue_variance_factor)
    prefix_weight = base_prefix_weight * decay_factor
if cache_hit_rate < LOW_THRESHOLD:
    prefix_weight = base_prefix_weight * 0.5  # don't chase cold caches
if kv_util_spread > IMBALANCE_THRESHOLD:
    kv_util_weight = base_kv_weight * (1 + spread_factor)

# Normalize weights to sum to 1.0
weights = normalize([prefix_weight, kv_util_weight, queue_weight])
```

**Enhancement 2: Queue threshold with deferred routing (from Dynamo)**
```
# Before scoring, check if ALL pods exceed queue threshold
if min(queue_depths) > QUEUE_THRESHOLD (default: 2.0):
    defer_ms = 10  # wait for fresher metrics
    # Re-fetch metrics after defer window
    # Route with updated state
```

This is high-impact because stale metrics cause pile-on: multiple requests see the same "low queue" pod and all route there simultaneously.

**Enhancement 3: Workload-aware weight profiles**
```
# Classify incoming request
if request.input_length > 10000:  # long-context
    profile = "long_context"  # prefix:4, kv:1, queue:3
elif request.has_system_prompt_prefix:
    profile = "conversational"  # prefix:5, kv:2, queue:2
else:
    profile = "default"  # prefix:3, kv:2, queue:2
```

### 3.3 Implementation plan

1. **Fork `llm-d-inference-scheduler`** — create branch `adaptive-scorer`
2. **Add `AdaptiveScorer` wrapper** — wraps existing scorers, adjusts weights per cycle
3. **Add `QueueThresholdFilter`** — new filter stage before scoring
4. **Add metrics export** — Prometheus metrics for weight values, defer counts
5. **Build + push custom image** — deploy to cluster
6. **A/B test:** static vs adaptive on Mooncake traces

### 3.4 Benchmark matrix

| Configuration | Weights | Queue Threshold | Deferred | Trace |
|---|---|---|---|---|
| Static default | 3:2:2 | No | No | conversation |
| Static GKE-tuned | 3:2:5 | No | No | conversation |
| Adaptive EMA | dynamic | No | No | conversation |
| Adaptive + defer | dynamic | 2.0 | Yes | conversation |
| Adaptive + defer | dynamic | 2.0 | Yes | toolagent |

**Target metrics to beat Dynamo gap:**
- TTFT P90 < 500ms (Dynamo claims sub-100ms but on GB200)
- Cache hit rate > 85%
- Queue variance < 1.0 across pods

---

## Stage 4: P/D Disaggregation (Hours 16-24)

### 4.1 Reconfigure cluster for P/D

**Topology:**
- **Prefill pool:** Node `g12540c` — 8 GPUs, TP=4, 2 replicas
  - Optimized for compute-bound first-token generation
  - Lower TP = less communication overhead for prefill
- **Decode pool:** Node `gf2ab44` — 8 GPUs, TP=2, 4 replicas
  - Optimized for memory-bound autoregressive decoding
  - Higher replica count for decode throughput

### 4.2 Deploy P/D components
- [ ] Deploy vLLM with `--kv-transfer-config` for disaggregation
- [ ] Deploy NIXL sidecar for RDMA-based KV cache transfer
- [ ] Configure EPP with P/D-aware routing:
  - Prefill scorer weights: `prefix:2, queue:1`
  - Decode scorer: route based on KV transfer affinity
  - Disaggregation threshold: 16 non-cached tokens triggers split
- [ ] Validate: request flows through prefill → KV transfer → decode

### 4.3 Benchmark P/D vs co-located
- [ ] Run Mooncake conversation trace against P/D topology
- [ ] Run Mooncake toolagent trace against P/D topology
- [ ] Compare: TTFT, ITL, total latency, throughput, KV transfer overhead
- [ ] Save as `pd_conversation.json`, `pd_toolagent.json`

---

## Stage 5: KV Cache Optimization Blueprint (Hours 24-30)

### 5.1 Blueprint scope (document, not full implementation)

Produce `/Users/yfama/CKS/kv-cache-blueprint.md` covering:

**Layer 1: Precise Prefix Cache Routing (already in llm-d)**
- Global distributed cache index via KV events
- Block-level hash matching (512-token blocks)
- Session stickiness for multi-turn conversations

**Layer 2: Hierarchical KV Cache Tiering (Dynamo-inspired)**
- Tier 0: GPU HBM (hot) — active decoding sequences
- Tier 1: CPU DRAM (warm) — recent prefix blocks, Mooncake-style pooling
- Tier 2: NVMe SSD (cold) — long-tail prefix blocks
- Tier 3: Networked storage (archive) — historical KV for rare prefixes
- Eviction policy: LRU with frequency boost (LFU hybrid)

**Layer 3: KV Cache Transfer Optimization**
- NIXL over RDMA for P/D transfers (already in llm-d)
- Layer-wise pipelining: overlap compute with KV transfer (Mooncake technique)
- Async prefetch: predict next decode node and pre-stage KV blocks

**Layer 4: Distributed KV Cache Pool (Mooncake architecture)**
- Master service for metadata/allocation/eviction
- Peer-to-peer block replication for hot prefixes
- Two-phase Put to prevent dirty reads
- Integration path: vLLM connector → Mooncake Store API

### 5.2 Measurements to capture
- [ ] KV cache hit rate vs request pattern (conversation vs toolagent)
- [ ] Cache index metadata overhead (should be ~1M:1 data-to-metadata)
- [ ] KV transfer latency between nodes (RDMA baseline)
- [ ] Memory utilization curves under sustained load

---

## Stage 6: Analysis & Report (Hours 30-36)

### 6.1 Comparative results table
```
| Metric          | Random | Static EPP | Adaptive EPP | P/D Disagg | P/D + Adaptive |
|-----------------|--------|-----------|-------------|-----------|---------------|
| Throughput      |        |           |             |           |               |
| TTFT P50        |        |           |             |           |               |
| TTFT P90        |        |           |             |           |               |
| TTFT P99        |        |           |             |           |               |
| ITL P90         |        |           |             |           |               |
| Cache Hit Rate  |        |           |             |           |               |
| Queue Variance  |        |           |             |           |               |
```

### 6.2 Deliverables
1. **Benchmark data** — JSON results for all configurations
2. **Adaptive scorer PR** — fork of `llm-d-inference-scheduler` with EMA weights + queue threshold
3. **KV cache blueprint** — architecture document with implementation roadmap
4. **Mooncake trace adapter** — reusable script for trace replay testing
5. **Gap analysis** — llm-d vs Dynamo with specific recommendations

---

## Time Budget Summary

| Stage | Hours | Priority | Blocking? |
|-------|-------|----------|-----------|
| 0: Prerequisites | 0-1 | P0 | Yes — GPU availability |
| 1: Deploy + baseline | 1-5 | P0 | Yes — foundation |
| 2: Mooncake traces | 5-8 | P0 | Yes — data for scoring tests |
| 3: Adaptive EPP scorer | 8-16 | P0 | No — core deliverable |
| 4: P/D disaggregation | 16-24 | P1 | No — can run in parallel with Stage 3 analysis |
| 5: KV cache blueprint | 24-30 | P1 | No — document, can draft offline |
| 6: Analysis & report | 30-36 | P0 | No — synthesis |

**Critical path:** Stages 0 → 1 → 2 → 3 (adaptive scorer is the highest-value deliverable)

**If time runs short:** Cut Stage 4 P/D testing and focus on adaptive EPP + blueprint document. The KV cache blueprint (Stage 5) can be completed offline since it's a design document.

---

## Verification Plan

1. **Deployment health:** `kubectl get pods -n kpouget-dev` — all pods Running, no restarts
2. **Inference validation:** curl completion request through gateway, verify coherent response
3. **EPP routing:** check Prometheus metrics for cache hit distribution across pods
4. **Adaptive scorer:** compare weight telemetry — weights should shift under varying load
5. **P/D validation:** verify KV transfer logs showing RDMA block movement between nodes
6. **Benchmark reproducibility:** each benchmark run 3x, report mean + stddev
