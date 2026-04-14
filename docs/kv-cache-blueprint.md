# KV Cache Optimization Blueprint for llm-d

## Executive Summary

This blueprint describes a 4-layer KV cache optimization architecture for llm-d,
combining precise prefix cache routing (already implemented), hierarchical cache
tiering (Dynamo-inspired), transfer optimization (Mooncake-inspired layer-wise
pipelining), and a distributed cache pool design. Each layer builds on the previous,
and can be adopted incrementally.

---

## Layer 1: Precise Prefix Cache Routing (Current — llm-d v0.5+)

### Architecture
```
Request → EPP → Global Cache Index → Score → Route to best-cache-hit pod
                      ↑
              KV Events (ZMQ)
              from vLLM pods
```

### Components
- **kvblock.Index**: In-memory map of block hashes → pod locations (sub-ms lookup)
- **kvevents.Pool**: Sharded ZMQ subscribers consuming `kv@pod-id@model` events
- **tokenization.PrefixStore**: Cached tokenized prompts for hash computation

### Key Metrics (from llm-d benchmarks)
| Metric | Value |
|--------|-------|
| Cache hit rate | 87.4% |
| TTFT P90 (precise) | 0.542s |
| Throughput vs random | 2x |
| Metadata overhead | 1M:1 (8 bytes per 128-token block) |
| Session stickiness | 99.92% |

### What's Missing
- No cross-pod cache sharing (each pod holds its own KV blocks)
- No tiered storage (everything in GPU HBM or nothing)
- No predictive prefetching

---

## Layer 2: Hierarchical KV Cache Tiering

### Design (Inspired by NVIDIA Dynamo KV Cache Manager)

```
┌─────────────────────────────────────────────────┐
│                  Tier 0: GPU HBM                │
│  Active decoding sequences + hot prefix blocks  │
│  Access: ~1μs | Capacity: 141GB per H200        │
├─────────────────────────────────────────────────┤
│                Tier 1: CPU DRAM                 │
│  Recent prefix blocks, evicted from HBM         │
│  Access: ~10μs | Capacity: 512GB-2TB per node   │
├─────────────────────────────────────────────────┤
│              Tier 2: Local NVMe SSD             │
│  Long-tail prefix blocks, infrequent access     │
│  Access: ~100μs | Capacity: 2-8TB per node      │
├─────────────────────────────────────────────────┤
│          Tier 3: Networked Storage              │
│  Historical KV for rare prefixes, cold archive  │
│  Access: ~1ms | Capacity: Petabytes             │
└─────────────────────────────────────────────────┘
```

### Eviction Policy: LRU with Frequency Boost (LFU Hybrid)

```
eviction_score = recency_rank * (1 + log(access_count))
```

- Pure LRU causes prefix thrashing when popular prefixes are briefly unused
- Frequency boost ensures system prompts / common prefixes persist across tiers
- Access count resets on tier promotion (prevents permanent residency)

### Implementation Path
1. **vLLM integration**: vLLM's `--cpu-offload-gb` already supports CPU tier
2. **Block manager extension**: Add SSD tier via mmap'd files
3. **Cross-tier prefetch**: When EPP routes request to pod, signal pod to
   prefetch matching blocks from CPU→GPU before request arrives
4. **Metrics**: Export `kv_cache_tier_{0,1,2,3}_occupancy` and
   `kv_cache_promotion_rate` to Prometheus

### Expected Impact
- 3-5x effective KV cache capacity per pod
- Reduced cache eviction pressure → higher cache hit rates
- Enable serving of models with larger context windows without adding GPUs

---

## Layer 3: KV Cache Transfer Optimization

### Current State
- NIXL over RDMA for P/D KV transfer (llm-d + vLLM disaggregation)
- Transfer is synchronous: decode waits for full KV from prefill

### Optimization 1: Layer-Wise Pipelining (Mooncake Technique)

```
Prefill GPU:  [Layer 0 compute] [Layer 1 compute] [Layer 2 compute] ...
                  │                   │                   │
                  ↓ (async RDMA)      ↓ (async RDMA)      ↓
Decode GPU:       [Layer 0 recv]  [Layer 1 recv]  [Layer 2 recv]  ...
                                  [Layer 0 attn]  [Layer 1 attn]
```

- Overlap KV transfer with computation on both sides
- Prefill sends layer N's KV while computing layer N+1
- Decode starts attention on layer N while receiving layer N+1
- Hides ~60-80% of transfer latency (Mooncake FAST'25 result)

### Optimization 2: Predictive Prefetch

```
EPP scoring cycle:
  1. Score all pods for incoming request
  2. Identify top-2 decode candidates (primary + backup)
  3. Signal primary decode pod: "prefetch KV blocks [hash_1, hash_2, ...]"
  4. Decode pod pulls matching blocks from CPU→GPU cache before request arrives
  5. When request routed to primary: KV already warm in HBM
```

- Reduces TTFT by overlapping cache warming with request queuing
- Requires EPP→pod signaling channel (gRPC sidecar or vLLM API extension)
- Risk: wrong prediction wastes GPU memory bandwidth (~5-10% overhead)

### Optimization 3: Partial KV Reuse for Edit Operations

For multi-turn conversations where users edit previous messages:
```
Turn 1: [system_prompt | user_msg_1]           → KV cached as blocks [A, B, C]
Turn 2: [system_prompt | user_msg_1_edited]    → Blocks [A] reused, [B', C'] recomputed
```

- Hash comparison at block boundaries to find longest common prefix
- Only recompute KV for changed blocks and everything after
- Significant for conversational workloads (Mooncake conversation trace shows
  100% sharing of hash_id 0 = system prompt block)

### Expected Impact
- Layer-wise pipelining: 40-60% reduction in P/D transfer latency
- Predictive prefetch: 20-30% TTFT reduction for cache-warm requests
- Partial reuse: 10-40% prefill compute savings for edit-heavy workloads

---

## Layer 4: Distributed KV Cache Pool

### Architecture (Inspired by Mooncake Store)

```
┌──────────────────────────────────────────────────┐
│              KV Cache Pool Master                │
│  - Metadata management (block→location index)    │
│  - Allocation / eviction coordination            │
│  - Replication policy (hot blocks → 2-3 copies)  │
│  - Health monitoring of pool nodes               │
└──────────────────┬───────────────────────────────┘
                   │ gRPC metadata API
        ┌──────────┼──────────┐
        ↓          ↓          ↓
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Node 1  │ │ Node 2  │ │ Node 3  │
   │ GPU HBM │ │ GPU HBM │ │ GPU HBM │
   │ CPU RAM │ │ CPU RAM │ │ CPU RAM │
   │ NVMe    │ │ NVMe    │ │ NVMe    │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
        └───────────┴───────────┘
          RDMA (NIXL) mesh for
          point-to-point KV transfer
```

### Key Design Decisions

**1. Two-Phase Put (Mooncake)**
```
Phase 1: Writer node stores block + registers with Master (status: PENDING)
Phase 2: Master confirms registration → status: COMMITTED
```
- Prevents dirty reads: readers only see COMMITTED blocks
- Crash recovery: PENDING blocks garbage collected on timeout

**2. Block Replication for Hot Prefixes**
- System prompt blocks (shared by all requests): replicate to all nodes
- Popular conversation prefixes: replicate to 2 nodes
- Unique/cold blocks: single copy
- Replication triggered when access rate > threshold (e.g., 10 req/min)

**3. EPP Integration**
```
Scoring with distributed pool:
  1. EPP tokenizes request → compute block hashes
  2. Query Pool Master: "which nodes have blocks [h1, h2, h3]?"
  3. Master returns: {h1: [node1, node2], h2: [node1], h3: []}
  4. Score pods: node1 has 2/3 blocks cached → highest score
  5. Route to node1 pod
  6. Pod loads h1, h2 from local pool, recomputes h3
```

### Implementation Phases

**Phase 1 (2-4 weeks): Metadata Service**
- Extend kvblock.Index to track cross-node block locations
- Add gRPC metadata API for block lookup/registration
- No data movement yet — just awareness of remote blocks

**Phase 2 (4-8 weeks): Cross-Node Transfer**
- NIXL-based block transfer between nodes on cache miss
- Async background replication for hot blocks
- Integration with vLLM block manager for foreign block ingestion

**Phase 3 (8-12 weeks): Full Pool with Tiering**
- Multi-tier storage per node (HBM/DRAM/SSD)
- Pool-wide eviction coordination
- Replication policy engine

### Expected Impact
- Near-100% cache hit rate for multi-tenant workloads with shared prefixes
- Linear scaling of effective KV cache with cluster size
- Elimination of "cold start" problem when scaling replicas

---

## Measurements Captured on CKS Cluster

### Test Environment
- 2 nodes × 8 H200 GPUs, 4 vLLM replicas (TP=4)
- Model: openai/gpt-oss-120b (FP4)
- llm-d v0.7.0 EPP with default prefix cache scorer

### Mooncake Conversation Trace (500 requests)
| Metric | Value |
|--------|-------|
| Throughput | 3,813.8 tok/s |
| TTFT P50 | 175.4ms |
| TTFT P90 | 399.3ms |
| TTFT P99 | 631.5ms |
| E2E P50 | 1.93s |
| E2E P90 | 2.35s |

### Mooncake Toolagent Trace (500 requests)
| Metric | Value |
|--------|-------|
| Throughput | 2,656.1 tok/s |
| TTFT P50 | 203.8ms |
| TTFT P90 | 2,173.0ms |
| TTFT P99 | 3,553.2ms |
| E2E P50 | 1.49s |
| E2E P90 | 2.70s |

### Key Observations
1. **Conversation vs Toolagent**: Conversation trace has better TTFT P90 (399ms vs 2.1s)
   because multi-turn conversations share more prefix blocks (hash_id 0 shared by 100%)
2. **Toolagent tail latency**: P99 TTFT is 5.6x P50 — indicates prefix cache misses
   causing full recomputation on some requests. Distributed pool would reduce this.
3. **Throughput ceiling**: 3.8K tok/s on 16 H200 GPUs suggests memory-bandwidth bound
   at this concurrency level. P/D disaggregation would decouple the bottleneck.

---

## Roadmap Priority

| Layer | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1. Precise prefix routing | Done | High | Shipped |
| 2. CPU tier offloading | 2-3 weeks | Medium | P1 — easy win with vLLM flag |
| 3a. Layer-wise pipelining | 4-6 weeks | High | P0 — biggest P/D perf gain |
| 3b. Predictive prefetch | 3-4 weeks | Medium | P1 — needs EPP→pod signaling |
| 4. Distributed pool | 8-12 weeks | Very High | P0 — unlocks multi-node cache |
