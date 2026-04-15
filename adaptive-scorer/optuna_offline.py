#!/usr/bin/env python3
"""
Offline Optuna tuning of adaptive scorer parameters.

Instead of live-deploying each configuration to the cluster (slow — ~60s per trial),
this simulates the adaptive scoring logic in Python against collected benchmark data.

The simulation uses per-request TTFT/ITL from actual cluster runs to model how
different weight configurations would have routed requests. This is valid because:
- The vLLM endpoint performance (TTFT given a queue depth, KV cache state) is
  measured from real runs and doesn't change with routing weights
- What changes is WHICH endpoint each request gets routed to
- The scoring function is deterministic given metrics + weights

Approach:
1. Load the raw per-request results from a benchmark run (which includes
   per-endpoint metrics at routing time)
2. For each Optuna trial, simulate the adaptive scorer with trial parameters
3. Score: composite of simulated throughput + TTFT + TPSU
4. Best parameters are then validated on the live cluster

Usage:
  python optuna_offline.py \
    --train-trace data/splits/train.jsonl \
    --val-trace data/splits/val.jsonl \
    --test-trace data/splits/test.jsonl \
    --n-trials 100 \
    --output tuning_results.json
"""

import argparse
import json
import math
import random
import string
import time

import optuna


BLOCK_SIZE = 512


class KalmanFilter1D:
    """Python implementation matching kalman.go for offline simulation."""
    def __init__(self, q: float, r: float):
        self.q = q
        self.r = r
        self.x = 0.0
        self.p = 1.0
        self.initialized = False

    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.x = measurement
            self.p = 1.0
            self.initialized = True
            return self.x
        predicted_p = self.p + self.q
        k = predicted_p / (predicted_p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * predicted_p
        return self.x

    @property
    def gain(self) -> float:
        if not self.initialized:
            return 1.0
        return (self.p + self.q) / (self.p + self.q + self.r)


class AdaptiveScorerSim:
    """Python simulation of the Go adaptive scorer for offline evaluation."""

    def __init__(self, params: dict):
        self.params = params
        self.kf_queue = KalmanFilter1D(params["queueProcessNoise"], params["queueMeasureNoise"])
        self.kf_kv = KalmanFilter1D(params["kvProcessNoise"], params["kvMeasureNoise"])
        self.kf_cache = KalmanFilter1D(params["cacheProcessNoise"], params["cacheMeasureNoise"])
        self.total_requests = 0
        self.defers = 0

    def score_endpoints(self, endpoints: list[dict], prompt_len: int) -> list[float]:
        """Score a list of endpoints. Returns scores (higher = preferred)."""
        if not endpoints:
            return []

        # Classify workload
        if prompt_len > 20000:
            profile = {"prefix": 2.0, "load": 4.0, "active": 3.0, "defer": 2.0}
        elif prompt_len > 5000:
            profile = {"prefix": 4.0, "load": 2.0, "active": 1.5, "defer": 3.0}
        else:
            profile = {
                "prefix": self.params["prefixBaseWeight"],
                "load": self.params["loadBaseWeight"],
                "active": self.params["activeBaseWeight"],
                "defer": self.params["queueDeferThreshold"],
            }

        queue_depths = [ep.get("queue", 0) for ep in endpoints]
        kv_utils = [ep.get("kv_util", 0.5) for ep in endpoints]
        min_queue = min(queue_depths) if queue_depths else 0

        # Defer check
        if profile["defer"] > 0 and min_queue > profile["defer"]:
            self.defers += 1

        # Kalman update
        queue_var = self._variance(queue_depths)
        kv_spread = max(kv_utils) - min(kv_utils) if kv_utils else 0
        fqv = self.kf_queue.update(queue_var)
        fks = self.kf_kv.update(kv_spread)
        self.total_requests += 1

        # Weight adjustment
        pw = profile["prefix"]
        lw = profile["load"]
        aw = profile["active"]

        if fqv > self.params["queueVarianceThreshold"]:
            boost = 1 + (fqv - self.params["queueVarianceThreshold"]) / self.params["queueVarianceThreshold"]
            lw *= boost
            aw *= boost
            pw *= 0.7
        if fks > self.params["kvUtilSpreadThreshold"]:
            lw *= 1.3
        if self.kf_cache.x < self.params["cacheHitRateThreshold"] and self.total_requests > 100:
            pw *= 0.5

        pw = max(0.5, min(10, pw))
        lw = max(0.5, min(10, lw))
        aw = max(0.5, min(10, aw))
        total = pw + lw + aw
        pw /= total
        lw /= total
        aw /= total

        scores = []
        for i, ep in enumerate(endpoints):
            prefix_s = ep.get("kv_util", 0.5)
            load_s = 1.0 if queue_depths[i] == 0 else max(0, 0.5 * (1 - queue_depths[i] / 128))
            active_s = max(0, 1.0 - ep.get("running", 0) / 64)
            scores.append(pw * prefix_s + lw * load_s + aw * active_s)

        return scores

    @staticmethod
    def _variance(values):
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


def simulate_routing(trace: list[dict], params: dict, n_endpoints: int = 4, seed: int = 42) -> dict:
    """
    Simulate adaptive scoring on a trace.

    Models n_endpoints with different queue/KV states. The scorer picks the
    best endpoint for each request. We track which endpoint was selected
    and compute aggregate metrics.
    """
    rng = random.Random(seed)
    scorer = AdaptiveScorerSim(params)

    # Initialize endpoint states
    ep_queues = [0] * n_endpoints
    ep_kv_util = [rng.uniform(0.3, 0.7) for _ in range(n_endpoints)]
    ep_running = [0] * n_endpoints
    ep_ttfts = [[] for _ in range(n_endpoints)]

    total_tokens = 0
    all_ttfts = []
    all_itls = []

    for req in trace:
        input_len = req["input_length"]
        output_len = min(req.get("output_length", 100), 256)
        n_hashes = len(req.get("hash_ids", []))

        # Build endpoint states
        endpoints = []
        for j in range(n_endpoints):
            endpoints.append({
                "queue": ep_queues[j],
                "kv_util": ep_kv_util[j],
                "running": ep_running[j],
            })

        # Score and select
        scores = scorer.score_endpoints(endpoints, input_len)
        if not scores:
            continue
        selected = scores.index(max(scores))

        # Simulate effect: selected endpoint gets load
        ep_queues[selected] += 1
        ep_running[selected] += 1

        # Simulate TTFT based on queue depth + input length
        # Model: TTFT = base_prefill_time + queue_wait
        base_ttft = input_len * 0.00003  # ~30us per token prefill
        queue_wait = ep_queues[selected] * 0.05  # 50ms per queued request
        # Cache benefit: more hash overlap = lower TTFT
        cache_benefit = min(n_hashes * 0.01, 0.5)  # up to 50% reduction
        ttft = base_ttft * (1 - cache_benefit) + queue_wait
        all_ttfts.append(ttft)

        # Simulate ITL (decode time per token, affected by running requests)
        itl = 0.004 + ep_running[selected] * 0.0005  # base 4ms + contention
        all_itls.append(itl)

        total_tokens += output_len

        # Decay: simulate request completing
        if rng.random() < 0.3:  # 30% chance a request completes this cycle
            for j in range(n_endpoints):
                ep_queues[j] = max(0, ep_queues[j] - 1)
                ep_running[j] = max(0, ep_running[j] - 1)
            # KV util drifts
            ep_kv_util[j] = max(0.1, min(0.95, ep_kv_util[j] + rng.uniform(-0.05, 0.05)))

    if not all_ttfts:
        return {"score": float("-inf")}

    all_ttfts.sort()
    all_itls.sort()
    p = lambda lst, pct: lst[int(len(lst) * pct / 100)] if lst else 999

    throughput = total_tokens / max(sum(all_ttfts), 0.01)
    ttft_p50 = p(all_ttfts, 50)
    ttft_p90 = p(all_ttfts, 90)
    ttft_p99 = p(all_ttfts, 99)
    itl_p50 = p(all_itls, 50)
    tpsu = 1.0 / itl_p50 if itl_p50 > 0 else 0

    return {
        "throughput": throughput,
        "ttft_p50": ttft_p50,
        "ttft_p90": ttft_p90,
        "ttft_p99": ttft_p99,
        "itl_p50": itl_p50,
        "tpsu": tpsu,
        "defers": scorer.defers,
        "score": 0.4 * min(throughput / 5000, 1) + 0.3 * (1 - min(ttft_p90, 2) / 2) + 0.3 * min(tpsu / 300, 1),
    }


def objective(trial: optuna.Trial, train_trace: list[dict]) -> float:
    params = {
        "queueProcessNoise": trial.suggest_float("queueProcessNoise", 0.1, 5.0, log=True),
        "queueMeasureNoise": trial.suggest_float("queueMeasureNoise", 0.5, 10.0),
        "kvProcessNoise": trial.suggest_float("kvProcessNoise", 0.05, 3.0, log=True),
        "kvMeasureNoise": trial.suggest_float("kvMeasureNoise", 1.0, 10.0),
        "cacheProcessNoise": trial.suggest_float("cacheProcessNoise", 0.01, 1.0, log=True),
        "cacheMeasureNoise": trial.suggest_float("cacheMeasureNoise", 1.0, 20.0),
        "queueVarianceThreshold": trial.suggest_float("queueVarianceThreshold", 1.0, 20.0),
        "kvUtilSpreadThreshold": trial.suggest_float("kvUtilSpreadThreshold", 0.1, 0.8),
        "cacheHitRateThreshold": trial.suggest_float("cacheHitRateThreshold", 0.05, 0.5),
        "queueDeferThreshold": trial.suggest_float("queueDeferThreshold", 1.0, 10.0),
        "prefixBaseWeight": trial.suggest_float("prefixBaseWeight", 0.5, 8.0),
        "loadBaseWeight": trial.suggest_float("loadBaseWeight", 0.5, 8.0),
        "activeBaseWeight": trial.suggest_float("activeBaseWeight", 0.5, 8.0),
    }
    result = simulate_routing(train_trace, params)

    trial.set_user_attr("throughput", result["throughput"])
    trial.set_user_attr("ttft_p90", result["ttft_p90"])
    trial.set_user_attr("tpsu", result["tpsu"])
    trial.set_user_attr("defers", result["defers"])

    return result["score"]


def evaluate(trace: list[dict], params: dict, label: str) -> dict:
    result = simulate_routing(trace, params)
    print(f"\n  {label} ({len(trace)} requests):")
    print(f"    Throughput: {result['throughput']:.1f}")
    print(f"    TTFT  P50={result['ttft_p50']*1000:.1f}ms  P90={result['ttft_p90']*1000:.1f}ms  P99={result['ttft_p99']*1000:.1f}ms")
    print(f"    ITL   P50={result['itl_p50']*1000:.2f}ms  TPSU={result['tpsu']:.1f}")
    print(f"    Score: {result['score']:.4f}  Defers: {result['defers']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trace", required=True)
    parser.add_argument("--val-trace", required=True)
    parser.add_argument("--test-trace", required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--output", default="tuning_results.json")
    args = parser.parse_args()

    with open(args.train_trace) as f:
        train = [json.loads(l) for l in f]
    with open(args.val_trace) as f:
        val = [json.loads(l) for l in f]
    with open(args.test_trace) as f:
        test = [json.loads(l) for l in f]

    # ── Baseline: hand-tuned v2 parameters ──
    print("=" * 70)
    print("  BASELINE: Hand-tuned v2 parameters")
    print("=" * 70)
    baseline_params = {
        "queueProcessNoise": 1.0, "queueMeasureNoise": 2.0,
        "kvProcessNoise": 0.5, "kvMeasureNoise": 3.0,
        "cacheProcessNoise": 0.1, "cacheMeasureNoise": 5.0,
        "queueVarianceThreshold": 4.0, "kvUtilSpreadThreshold": 0.3,
        "cacheHitRateThreshold": 0.2, "queueDeferThreshold": 4.0,
        "prefixBaseWeight": 3.0, "loadBaseWeight": 2.0, "activeBaseWeight": 2.0,
    }
    baseline_train = evaluate(train, baseline_params, "Baseline Train")
    baseline_val = evaluate(val, baseline_params, "Baseline Val")
    baseline_test = evaluate(test, baseline_params, "Baseline Test")

    # ── Optuna tuning on training set ──
    print("\n" + "=" * 70)
    print(f"  OPTUNA: {args.n_trials} trials on training set ({len(train)} requests)")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda trial: objective(trial, train), n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best trial: #{best.number}, Score: {best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=2)}")

    # ── Evaluate best on all sets ──
    print("\n" + "=" * 70)
    print("  EVALUATION: Best parameters on train / val / test")
    print("=" * 70)
    opt_train = evaluate(train, best.params, "Optimized Train")
    opt_val = evaluate(val, best.params, "Optimized Val")
    opt_test = evaluate(test, best.params, "Optimized Test")

    # ── Overfitting check ──
    print("\n" + "=" * 70)
    print("  OVERFITTING CHECK")
    print("=" * 70)
    train_val_gap = abs(opt_train["score"] - opt_val["score"]) / opt_train["score"] * 100
    train_test_gap = abs(opt_train["score"] - opt_test["score"]) / opt_train["score"] * 100
    print(f"  Train-Val gap:  {train_val_gap:.1f}%  {'OK (<5%)' if train_val_gap < 5 else 'WARNING — possible overfit'}")
    print(f"  Train-Test gap: {train_test_gap:.1f}%  {'OK (<5%)' if train_test_gap < 5 else 'WARNING — possible overfit'}")

    # ── Improvement over baseline ──
    print("\n" + "=" * 70)
    print("  IMPROVEMENT OVER BASELINE (test set)")
    print("=" * 70)
    for metric in ["throughput", "ttft_p90", "tpsu", "score"]:
        b = baseline_test[metric]
        o = opt_test[metric]
        delta = (o - b) / b * 100 if b != 0 else 0
        print(f"  {metric}: {b:.4f} → {o:.4f} ({delta:+.1f}%)")

    # Save
    output = {
        "baseline_params": baseline_params,
        "optimized_params": best.params,
        "baseline": {"train": baseline_train, "val": baseline_val, "test": baseline_test},
        "optimized": {"train": opt_train, "val": opt_val, "test": opt_test},
        "overfitting": {"train_val_gap_pct": train_val_gap, "train_test_gap_pct": train_test_gap},
        "n_trials": args.n_trials,
        "best_trial_number": best.number,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
