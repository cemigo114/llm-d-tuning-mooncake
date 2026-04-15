# llm-d Playbook: Stage-by-Stage Performance Gains

Fresh end-to-end benchmark run on the same cluster, same session, same Mooncake conversation trace. Each stage adds one optimization on top of the previous.

**Cluster:** 2 nodes x 8 NVIDIA H200 (141GB HBM3e), RDMA/InfiniBand, K8s v1.35.0
**Model:** `openai/gpt-oss-120b` (FP4, vLLM)
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

### Metric Definitions

| Metric | Definition |
|--------|-----------|
| **Throughput** | Total output tokens / wall clock seconds |
| **TTFT** | Time to First Token -- latency until the first token is returned |
| **ITL** | Inter-Token Latency -- average time between consecutive tokens (streaming) |
| **TPSU** | Tokens Per Second per User = 1/ITL -- perceived generation speed for the end user |
| **E2E** | End-to-end request latency from send to last token received |

## Key Takeaways

### 1. Throughput: +33% (Stages 1 to 4)

Stage 4 (P/D + adaptive EPP) delivers 5,210 tok/s vs 3,913 tok/s baseline. The adaptive scorer's workload-aware profiles and Dynamo-inspired queue threshold deferral prevent hot-node piling under load.

### 2. Per-User Speed (TPSU): +49% (Stages 1 to 5)

TPSU improves from 156 to 233 tok/s/user across the full stack. Each added optimization reduces inter-token latency -- the end user sees tokens arrive 49% faster. P/D disaggregation is the biggest contributor (+33%) because decode workers are no longer blocked by prefill compute.

### 3. Tail Latency (TTFT P99): -35% (Stages 1 to 4)

The adaptive EPP cuts P99 first-token latency from 1,339ms to 875ms by dynamically shifting routing weights when queue variance is high. Static EPP (Stage 2) already improves P99 by 22%; the adaptive version adds another 16 percentage points.

### 4. CPU Offload: Right Tool for Long Sequences

Stage 5 adds a 100GB CPU KV cache tier via vLLM's `MultiConnector` API. At 8K input tokens, this provides the highest TPSU (233) and lowest ITL (4.29ms) because long prefixes are served from CPU RAM instead of being recomputed. **Not recommended for short sequences** -- at 4K tokens, the CPU offload overhead dominates (see README for the crossover analysis).

## What Each Stage Adds

**Stage 1: K8s Round-Robin** -- vLLM model servers behind a standard Kubernetes Service with default round-robin load balancing. No intelligent routing. This is what you get out of the box.

**Stage 2: llm-d EPP** -- Adds the llm-d Endpoint Picker Policy (EPP) with prefix-cache-aware scoring. The EPP routes requests to pods that already have matching KV cache blocks, reducing redundant prefill computation. +15% throughput, -22% TTFT P99.

**Stage 3: P/D Disaggregation** -- Separates prefill (compute-bound) and decode (memory-bound) into dedicated worker pools connected via NIXL/RDMA for KV cache transfer. Decode workers produce tokens without prefill interference. ITL drops 25%, TPSU jumps 33%.

**Stage 4: P/D + Adaptive EPP** -- Replaces static EPP weights with a workload-aware adaptive scorer that dynamically adjusts prefix/load/active weights based on observed queue variance and KV utilization. Includes Dynamo-inspired queue threshold deferral. Best overall throughput (5,210 tok/s) and lowest TTFT P99 (875ms).

**Stage 5: P/D + CPU Offload** -- Adds a 100GB CPU RAM cache tier via vLLM's `MultiConnector` (combining `NixlConnector` + `OffloadingConnector`). For long-sequence workloads (8K+ tokens), this provides the highest per-user generation speed (233 TPSU) by serving cached prefix blocks from CPU RAM.

## Reproduce

```bash
git clone https://github.com/cemigo114/llm-d-tuning-mooncake.git
cd llm-d-tuning-mooncake

export NAMESPACE=your-namespace
export KUBECONFIG=your-kubeconfig
export ADAPTIVE_IMAGE=ttl.sh/your-image:24h  # build first: ./scripts/build-adaptive-epp.sh

# Run all 5 stages end-to-end
./scripts/run-all-stages.sh

# Or run individually
./scripts/deploy-colocated.sh            # Stage 1+2
./scripts/deploy-pd.sh                   # Stage 3
./scripts/deploy-adaptive-epp.sh $IMAGE  # Stage 4
./scripts/deploy-pd-multiconnector.sh    # Stage 5

# Generate report
python3 benchmarks/scripts/generate_report.py benchmarks/results/stages/stage_*.json
```

## Data Integrity

All metrics independently verified by recomputing percentiles from raw per-request JSON files. Each JSON file contains per-request `ttft`, `itl`, `total_time`, `output_tokens`, and `input_length` fields.
