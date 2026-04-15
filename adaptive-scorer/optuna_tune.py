#!/usr/bin/env python3
"""
Bayesian optimization of adaptive EPP scorer parameters using Optuna.

Tunes 8 parameters against a training trace, evaluates on validation,
and reports final metrics on a held-out test set.

The objective function runs the Mooncake replay benchmark with a given
parameter configuration and returns a composite score:

  score = throughput_weight * throughput_normalized
        + ttft_weight * (1 - ttft_p90_normalized)
        + itl_weight * (1 - itl_p50_normalized)

This maximizes throughput while minimizing TTFT and ITL.

Usage:
  python optuna_tune.py \
    --endpoint http://gateway:80 \
    --model openai/gpt-oss-120b \
    --train-trace data/splits/train.jsonl \
    --val-trace data/splits/val.jsonl \
    --test-trace data/splits/test.jsonl \
    --n-trials 50 \
    --output tuning_results.json

Prerequisites:
  pip install optuna aiohttp
"""

import argparse
import asyncio
import json
import math
import random
import string
import time
from typing import Optional

import aiohttp
import optuna


# ── Replay logic (inline to avoid import issues in cluster pods) ───────────

BLOCK_SIZE = 512
VOCAB = string.ascii_lowercase + " "


def hash_id_to_text(hash_id: int, tokens_per_block: int = BLOCK_SIZE) -> str:
    rng = random.Random(hash_id)
    return "".join(rng.choice(VOCAB) for _ in range(tokens_per_block * 4))


def build_prompt(hash_ids: list[int], max_input_tokens: int) -> str:
    blocks, total = [], 0
    for hid in hash_ids:
        bt = min(BLOCK_SIZE, max_input_tokens - total)
        if bt <= 0:
            break
        blocks.append(hash_id_to_text(hid, bt))
        total += bt
    return " ".join(blocks)


async def send_request(session, endpoint, model, d, max_input_tokens, results, sem):
    prompt = build_prompt(d["hash_ids"], max_input_tokens)
    async with sem:
        t0 = time.monotonic()
        ttft = None
        token_times = []
        output_tokens = 0
        try:
            async with session.post(
                f"{endpoint}/v1/completions",
                json={"model": model, "prompt": prompt,
                      "max_tokens": min(d["output_length"], 256),
                      "temperature": 0, "stream": True},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        now = time.monotonic()
                        if ttft is None:
                            ttft = now - t0
                        try:
                            c = json.loads(decoded[6:])
                            if c.get("choices", [{}])[0].get("text"):
                                output_tokens += 1
                                token_times.append(now)
                        except json.JSONDecodeError:
                            pass
            total_time = time.monotonic() - t0
            itl = None
            if len(token_times) >= 2:
                deltas = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
                itl = sum(deltas) / len(deltas)
            results.append({"ttft": ttft, "total_time": total_time,
                            "output_tokens": output_tokens, "itl": itl, "status": "ok"})
        except Exception:
            results.append({"status": "error"})


async def run_replay(trace_path, endpoint, model, max_requests, max_concurrency, max_input_tokens):
    with open(trace_path) as f:
        data = [json.loads(l) for l in f][:max_requests]
    sem = asyncio.Semaphore(max_concurrency)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        t0 = time.monotonic()
        for i, d in enumerate(data):
            if i > 0:
                delta = d["timestamp"] - data[i-1]["timestamp"]
                await asyncio.sleep(max(0, delta * 0.0005 / 1000))
            tasks.append(asyncio.create_task(
                send_request(session, endpoint, model, d, max_input_tokens, results, sem)))
        await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return results, wall


def compute_metrics(results, wall_time):
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return None
    ttfts = sorted([r["ttft"] for r in ok if r.get("ttft") is not None])
    itls = sorted([r["itl"] for r in ok if r.get("itl") is not None])
    tokens = sum(r.get("output_tokens", 0) for r in ok)
    p = lambda lst, pct: lst[int(len(lst) * pct / 100)] if lst else 999
    return {
        "throughput": tokens / wall_time if wall_time > 0 else 0,
        "ttft_p50": p(ttfts, 50),
        "ttft_p90": p(ttfts, 90),
        "ttft_p99": p(ttfts, 99),
        "itl_p50": p(itls, 50) if itls else 0.01,
        "tpsu": 1.0 / p(itls, 50) if itls and p(itls, 50) > 0 else 0,
        "ok_count": len(ok),
        "total_count": len(results),
    }


# ── Objective function ─────────────────────────────────────────────────────

def make_objective(endpoint, model, train_trace, max_requests, max_concurrency,
                   max_input_tokens, config_apply_fn):
    """Create an Optuna objective that benchmarks a parameter configuration."""

    def objective(trial: optuna.Trial) -> float:
        # Sample the 8 tunable parameters
        params = {
            "alpha": trial.suggest_float("alpha", 0.01, 0.5, log=True),
            "queueVarianceThreshold": trial.suggest_float("queueVarianceThreshold", 1.0, 20.0),
            "kvUtilSpreadThreshold": trial.suggest_float("kvUtilSpreadThreshold", 0.1, 0.8),
            "cacheHitRateThreshold": trial.suggest_float("cacheHitRateThreshold", 0.05, 0.5),
            "queueDeferThreshold": trial.suggest_float("queueDeferThreshold", 1.0, 10.0),
            "prefixBaseWeight": trial.suggest_float("prefixBaseWeight", 0.5, 8.0),
            "loadBaseWeight": trial.suggest_float("loadBaseWeight", 0.5, 8.0),
            "activeBaseWeight": trial.suggest_float("activeBaseWeight", 0.5, 8.0),
        }

        # Apply parameters to the running EPP (via ConfigMap hot-reload)
        config_apply_fn(params)
        time.sleep(2)  # Allow config propagation

        # Run benchmark
        results, wall = asyncio.run(
            run_replay(train_trace, endpoint, model, max_requests, max_concurrency, max_input_tokens)
        )
        metrics = compute_metrics(results, wall)
        if metrics is None:
            return float("-inf")

        # Composite objective: maximize throughput + TPSU, minimize TTFT P90
        # Normalize to comparable scales
        throughput_score = metrics["throughput"] / 5000  # ~1.0 at 5K tok/s
        ttft_penalty = min(metrics["ttft_p90"], 2.0) / 2.0  # [0,1], lower=better
        tpsu_score = min(metrics["tpsu"], 300) / 300  # ~1.0 at 300 tok/s/user

        score = 0.4 * throughput_score + 0.3 * (1 - ttft_penalty) + 0.3 * tpsu_score

        trial.set_user_attr("throughput", metrics["throughput"])
        trial.set_user_attr("ttft_p90", metrics["ttft_p90"])
        trial.set_user_attr("itl_p50", metrics["itl_p50"])
        trial.set_user_attr("tpsu", metrics["tpsu"])

        return score

    return objective


# ── Config application (writes to EPP ConfigMap) ──────────────────────────

def make_config_applier(namespace: str = "kpouget-dev"):
    """Returns a function that patches the EPP ConfigMap with new parameters."""
    import subprocess

    def apply_config(params: dict):
        config_yaml = f"""apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
featureGates:
- prepareDataPlugins
plugins:
- type: prefill-header-handler
- type: adaptive-scorer
  parameters:
    alpha: {params['alpha']}
    queueVarianceThreshold: {params['queueVarianceThreshold']}
    kvUtilSpreadThreshold: {params['kvUtilSpreadThreshold']}
    cacheHitRateThreshold: {params['cacheHitRateThreshold']}
    queueDeferThreshold: {params['queueDeferThreshold']}
    deferDuration: "10ms"
    prefixBaseWeight: {params['prefixBaseWeight']}
    loadBaseWeight: {params['loadBaseWeight']}
    activeBaseWeight: {params['activeBaseWeight']}
    enableWorkloadProfiles: true
- type: prefix-cache-scorer
  parameters:
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
- type: queue-scorer
- type: prefill-filter
- type: decode-filter
- type: max-score-picker
- type: prefix-based-pd-decider
  parameters:
    nonCachedTokens: 16
- type: pd-profile-handler
  parameters:
    primaryPort: 0
    deciderPluginName: prefix-based-pd-decider
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: max-score-picker
  - pluginRef: adaptive-scorer
    weight: 3
  - pluginRef: queue-scorer
    weight: 1
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: max-score-picker
  - pluginRef: adaptive-scorer
    weight: 3
  - pluginRef: queue-scorer
    weight: 1"""

        # Escape for JSON
        escaped = config_yaml.replace('"', '\\"').replace('\n', '\\n')
        patch = f'{{"data":{{"adaptive-pd-config.yaml":"{escaped}"}}}}'

        subprocess.run(
            ["kubectl", "patch", "configmap", "gaie-pd-epp",
             "-n", namespace, "--type", "merge", "-p", patch],
            capture_output=True, timeout=10
        )
        # Restart EPP to pick up config
        subprocess.run(
            ["kubectl", "rollout", "restart", "deployment/gaie-pd-epp",
             "-n", namespace],
            capture_output=True, timeout=10
        )
        # Wait for rollout
        subprocess.run(
            ["kubectl", "rollout", "status", "deployment/gaie-pd-epp",
             "-n", namespace, "--timeout=60s"],
            capture_output=True, timeout=70
        )

    return apply_config


def evaluate_on_set(trace_path, endpoint, model, max_requests, max_concurrency,
                    max_input_tokens, label):
    """Run benchmark on a trace set and print metrics."""
    results, wall = asyncio.run(
        run_replay(trace_path, endpoint, model, max_requests, max_concurrency, max_input_tokens)
    )
    metrics = compute_metrics(results, wall)
    if metrics:
        print(f"\n  {label} Set Results ({metrics['ok_count']}/{metrics['total_count']} ok):")
        print(f"    Throughput: {metrics['throughput']:.1f} tok/s")
        print(f"    TTFT P50={metrics['ttft_p50']*1000:.1f}ms  P90={metrics['ttft_p90']*1000:.1f}ms  P99={metrics['ttft_p99']*1000:.1f}ms")
        print(f"    ITL P50={metrics['itl_p50']*1000:.2f}ms  TPSU={metrics['tpsu']:.1f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for adaptive EPP scorer")
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--train-trace", required=True)
    parser.add_argument("--val-trace", required=True)
    parser.add_argument("--test-trace", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--namespace", default="kpouget-dev")
    parser.add_argument("--output", default="tuning_results.json")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip Optuna tuning, just evaluate with current params")
    args = parser.parse_args()

    config_applier = make_config_applier(args.namespace)

    if not args.skip_tuning:
        print("=" * 70)
        print("  PHASE 1: Bayesian Optimization on Training Set")
        print(f"  Trials: {args.n_trials}, Requests/trial: {args.max_requests}")
        print("=" * 70)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        objective = make_objective(
            args.endpoint, args.model, args.train_trace,
            args.max_requests, args.max_concurrency, args.max_input_tokens,
            config_applier,
        )

        study.optimize(objective, n_trials=args.n_trials)

        best = study.best_trial
        print(f"\n  Best trial: #{best.number}")
        print(f"  Score: {best.value:.4f}")
        print(f"  Params: {json.dumps(best.params, indent=2)}")
        print(f"  Throughput: {best.user_attrs.get('throughput', 'N/A')}")
        print(f"  TTFT P90: {best.user_attrs.get('ttft_p90', 'N/A')}")
        print(f"  TPSU: {best.user_attrs.get('tpsu', 'N/A')}")

        # Apply best params
        config_applier(best.params)
        time.sleep(5)
        best_params = best.params
    else:
        print("  Skipping tuning, using current parameters")
        best_params = {}

    # PHASE 2: Validate on validation set
    print("\n" + "=" * 70)
    print("  PHASE 2: Validation Set Evaluation")
    print("=" * 70)
    val_metrics = evaluate_on_set(
        args.val_trace, args.endpoint, args.model,
        args.max_requests, args.max_concurrency, args.max_input_tokens,
        "Validation"
    )

    # PHASE 3: Final evaluation on held-out test set
    print("\n" + "=" * 70)
    print("  PHASE 3: Held-Out Test Set Evaluation (FINAL)")
    print("=" * 70)
    test_metrics = evaluate_on_set(
        args.test_trace, args.endpoint, args.model,
        args.max_requests, args.max_concurrency, args.max_input_tokens,
        "Test"
    )

    # Save results
    output = {
        "best_params": best_params,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
