#!/usr/bin/env python3
"""
Unified benchmark script for llm-d playbook stages.
Computes TTFT, ITL, TPSU (1/ITL), E2E, and throughput from Mooncake traces.

Usage:
  python benchmark_stage.py \
    --trace conversation_trace.jsonl \
    --endpoint http://gateway:80 \
    --model openai/gpt-oss-120b \
    --stage "1-k8s-baseline" \
    --max-requests 500 \
    --max-concurrency 32 \
    --max-input-tokens 4096 \
    --output results/stage1.json
"""

import argparse
import asyncio
import json
import random
import string
import time
from typing import Optional

import aiohttp

BLOCK_SIZE = 512
VOCAB = string.ascii_lowercase + " "


def hash_id_to_text(hash_id: int, tokens_per_block: int = BLOCK_SIZE) -> str:
    rng = random.Random(hash_id)
    return "".join(rng.choice(VOCAB) for _ in range(tokens_per_block * 4))


def build_prompt(hash_ids: list[int], max_input_tokens: int) -> str:
    blocks = []
    total = 0
    for hid in hash_ids:
        bt = min(BLOCK_SIZE, max_input_tokens - total)
        if bt <= 0:
            break
        blocks.append(hash_id_to_text(hid, bt))
        total += bt
    return " ".join(blocks)


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    input_length: int,
    output_length: int,
    hash_ids: list[int],
    max_input_tokens: int,
    results: list,
    semaphore: asyncio.Semaphore,
):
    prompt = build_prompt(hash_ids, max_input_tokens)
    max_tokens = min(output_length, 256)

    async with semaphore:
        t0 = time.monotonic()
        ttft = None
        token_times = []
        output_tokens = 0

        try:
            async with session.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        now = time.monotonic()
                        if ttft is None:
                            ttft = now - t0
                        try:
                            chunk = json.loads(decoded[6:])
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("text"):
                                output_tokens += 1
                                token_times.append(now)
                        except json.JSONDecodeError:
                            pass

            total_time = time.monotonic() - t0

            # Compute ITL: average inter-token latency
            itl = None
            if len(token_times) >= 2:
                deltas = [
                    token_times[i] - token_times[i - 1]
                    for i in range(1, len(token_times))
                ]
                itl = sum(deltas) / len(deltas)

            results.append(
                {
                    "input_length": input_length,
                    "output_length": output_length,
                    "num_hash_ids": len(hash_ids),
                    "ttft": ttft,
                    "total_time": total_time,
                    "output_tokens": output_tokens,
                    "itl": itl,
                    "status": "ok",
                }
            )
        except Exception as e:
            results.append(
                {
                    "input_length": input_length,
                    "output_length": output_length,
                    "status": "error",
                    "error": str(e)[:200],
                }
            )


async def run_benchmark(
    trace_path: str,
    endpoint: str,
    model: str,
    stage: str,
    rate_scale: float,
    max_requests: int,
    max_concurrency: int,
    max_input_tokens: int,
    output_path: str,
):
    with open(trace_path) as f:
        trace_data = [json.loads(line) for line in f][:max_requests]

    print(f"\n{'='*70}")
    print(f"  STAGE: {stage}")
    print(f"  Trace: {trace_path.split('/')[-1]} ({len(trace_data)} requests)")
    print(f"  Endpoint: {endpoint}")
    print(f"  Max input tokens: {max_input_tokens}, Concurrency: {max_concurrency}")
    print(f"{'='*70}")

    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.monotonic()

        for i, d in enumerate(trace_data):
            if i > 0:
                delta_ms = d["timestamp"] - trace_data[i - 1]["timestamp"]
                await asyncio.sleep(max(0, delta_ms * rate_scale / 1000.0))

            tasks.append(
                asyncio.create_task(
                    send_request(
                        session,
                        endpoint,
                        model,
                        d["input_length"],
                        d["output_length"],
                        d["hash_ids"],
                        max_input_tokens,
                        results,
                        semaphore,
                    )
                )
            )

            if (i + 1) % 100 == 0:
                ok = sum(1 for r in results if r.get("status") == "ok")
                elapsed = time.monotonic() - start_time
                print(f"  Dispatched {i+1}/{len(trace_data)}, completed {ok}, {elapsed:.1f}s")

        await asyncio.gather(*tasks)

    wall_time = time.monotonic() - start_time
    ok_results = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]

    # Save raw results with stage metadata
    output = {
        "stage": stage,
        "trace": trace_path.split("/")[-1],
        "max_input_tokens": max_input_tokens,
        "max_concurrency": max_concurrency,
        "wall_time_s": round(wall_time, 2),
        "total_requests": len(results),
        "succeeded": len(ok_results),
        "failed": len(errors),
        "requests": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print report
    if not ok_results:
        print(f"  FAILED: 0/{len(results)} succeeded")
        return

    ttfts = sorted([r["ttft"] for r in ok_results if r.get("ttft") is not None])
    itls = sorted([r["itl"] for r in ok_results if r.get("itl") is not None])
    e2es = sorted([r["total_time"] for r in ok_results])
    total_out_tokens = sum(r.get("output_tokens", 0) for r in ok_results)
    throughput = total_out_tokens / wall_time

    p = lambda lst, pct: lst[int(len(lst) * pct / 100)] if lst else 0

    itl_p50 = p(itls, 50)
    tpsu_p50 = 1.0 / itl_p50 if itl_p50 > 0 else 0

    print(f"\n  Results: {len(ok_results)}/{len(results)} succeeded in {wall_time:.1f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  TTFT   P50={p(ttfts,50)*1000:.1f}ms  P90={p(ttfts,90)*1000:.1f}ms  P99={p(ttfts,99)*1000:.1f}ms")
    print(f"  ITL    P50={itl_p50*1000:.2f}ms  P90={p(itls,90)*1000:.2f}ms  P99={p(itls,99)*1000:.2f}ms")
    print(f"  TPSU   P50={tpsu_p50:.1f} tok/s  (1/ITL — tokens per second per user)")
    print(f"  E2E    P50={p(e2es,50):.2f}s  P90={p(e2es,90):.2f}s  P99={p(e2es,99):.2f}s")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="llm-d playbook stage benchmark")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--stage", required=True, help="Stage name for the report")
    parser.add_argument("--rate-scale", type=float, default=0.0005)
    parser.add_argument("--max-requests", type=int, default=500)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            trace_path=args.trace,
            endpoint=args.endpoint,
            model=args.model,
            stage=args.stage,
            rate_scale=args.rate_scale,
            max_requests=args.max_requests,
            max_concurrency=args.max_concurrency,
            max_input_tokens=args.max_input_tokens,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
