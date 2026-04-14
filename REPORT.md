# llm-d Playbook: Stage-by-Stage Performance Gains

Each stage adds optimization on top of the previous, measured on
the same cluster (2x8 H200, 16 GPUs) with Mooncake conversation trace.

| # | Stage | Throughput (tok/s) | TTFT P50 (ms) | TTFT P90 (ms) | ITL P50 (ms) | TPSU P50 (tok/s/user) | E2E P50 (s) |
| - | ----- | -----------------: | ------------: | ------------: | -----------: | --------------------: | ----------: |
| 1 | **1-k8s-roundrobin** | 3,913 (baseline) | 128  | 427 | 6  | 156.0 | 1.71 |
| 2 | **2-llmd-colocated-epp** | 4,517 (+15%) | 134 (+5%) | 511 | 6 (-10%) | 173.2 | 1.56 |
| 3 | **3-pd-disaggregation** | 4,200 (+7%) | 338 (+164%) | 698 | 5 (-25%) | 207.7 | 1.48 |
| 4 | **4-pd-adaptive-epp** | 5,209 (+33%) | 191 (+49%) | 507 | 5 (-26%) | 210.1 | 1.34 |
| 5 | **5-pd-multiconnector-cpu-offload** | 1,542 (-61%) | 172 (+34%) | 2785 | 4 (-33%) | 233.1 | 0.04 |

### Cumulative Gains (Stage 1 → Stage 5)

| Metric | Stage 1 (Baseline) | Stage 5 (Final) | Improvement |
| ------ | -----------------: | --------------: | ----------: |
| Throughput | 3,913 tok/s | 1,542 tok/s | -61% |
| TTFT P50 | 128 ms | 172 ms | +34% |
| TTFT P90 | 427 ms | 2785 ms | +552% |
| ITL P50 | 6 ms | 4 ms | -33% |
| TPSU P50 | 156.04793590579567 tok/s/user | 233.1000227689839 tok/s/user | +49% |
| E2E P50 | 1.71 s | 0.04 s | -98% |

### What Each Stage Adds

**Stage 1: 1-k8s-roundrobin**
- Requests: 500/500, Wall time: 26.2s
- File: `stage_1_k8s_roundrobin.json`

**Stage 2: 2-llmd-colocated-epp**
- Requests: 500/500, Wall time: 22.7s
- File: `stage_2_llmd_colocated.json`

**Stage 3: 3-pd-disaggregation**
- Requests: 500/500, Wall time: 24.4s
- File: `stage_3_pd_disagg.json`

**Stage 4: 4-pd-adaptive-epp**
- Requests: 500/500, Wall time: 19.7s
- File: `stage_4_pd_adaptive.json`

**Stage 5: 5-pd-multiconnector-cpu-offload**
- Requests: 300/300, Wall time: 12.6s
- File: `stage_5_pd_multiconnector.json`

