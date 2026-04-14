#!/usr/bin/env python3
"""
Mooncake trace replay adapter for llm-d benchmarking.

Converts Mooncake FAST25 trace data into OpenAI-compatible requests
with controlled prefix sharing to test EPP prefix-cache-aware routing.

Trace format:
  {"timestamp": int, "input_length": int, "output_length": int, "hash_ids": [int...]}

Usage:
  python mooncake_replay.py --trace conversation_trace.jsonl \
    --endpoint http://localhost:8080 \
    --model openai/gpt-oss-120b \
    --rate-scale 0.001 \
    --max-requests 500
"""

import argparse
import asyncio
import json
import time
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp


# Each hash_id maps to a deterministic 512-token text block.
# Requests sharing hash_ids will share prefix tokens, triggering
# prefix cache hits in vLLM's block manager.
BLOCK_SIZE = 512  # tokens per hash block (Mooncake default)
VOCAB = string.ascii_lowercase + " "


def hash_id_to_text(hash_id: int, tokens_per_block: int = BLOCK_SIZE) -> str:
    """Generate deterministic text block for a hash_id.
    Same hash_id always produces same text -> shared prefix in tokenizer."""
    rng = random.Random(hash_id)
    # ~4 chars per token approximation
    chars = tokens_per_block * 4
    return "".join(rng.choice(VOCAB) for _ in range(chars))


@dataclass
class TraceRequest:
    timestamp: int  # milliseconds
    input_length: int
    output_length: int
    hash_ids: list[int]


def load_trace(path: str, max_requests: Optional[int] = None) -> list[TraceRequest]:
    requests = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_requests and i >= max_requests:
                break
            d = json.loads(line)
            requests.append(TraceRequest(**d))
    return requests


def build_prompt(req: TraceRequest, max_input_tokens: int = 8192) -> str:
    """Build prompt with shared prefix blocks from hash_ids.
    Truncate to max_input_tokens to fit model context."""
    blocks = []
    total_tokens = 0
    for hid in req.hash_ids:
        block_tokens = min(BLOCK_SIZE, max_input_tokens - total_tokens)
        if block_tokens <= 0:
            break
        blocks.append(hash_id_to_text(hid, block_tokens))
        total_tokens += block_tokens
    return " ".join(blocks)


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    req: TraceRequest,
    max_input_tokens: int,
    results: list,
    semaphore: asyncio.Semaphore,
):
    prompt = build_prompt(req, max_input_tokens)
    max_tokens = min(req.output_length, 256)  # cap for benchmarking

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    async with semaphore:
        t0 = time.monotonic()
        ttft = None
        output_tokens = 0

        try:
            async with session.post(
                f"{endpoint}/v1/completions",
                json={**payload, "stream": True},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        try:
                            chunk = json.loads(decoded[6:])
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("text"):
                                output_tokens += 1
                        except json.JSONDecodeError:
                            pass

            total_time = time.monotonic() - t0
            results.append({
                "timestamp": req.timestamp,
                "input_length": req.input_length,
                "output_length": req.output_length,
                "num_hash_ids": len(req.hash_ids),
                "ttft": ttft,
                "total_time": total_time,
                "output_tokens": output_tokens,
                "itl": (total_time - (ttft or 0)) / max(output_tokens - 1, 1) if output_tokens > 0 else None,
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "timestamp": req.timestamp,
                "input_length": req.input_length,
                "output_length": req.output_length,
                "status": "error",
                "error": str(e),
            })


async def replay(
    trace_path: str,
    endpoint: str,
    model: str,
    rate_scale: float = 0.001,
    max_requests: Optional[int] = None,
    max_concurrency: int = 64,
    max_input_tokens: int = 8192,
    output_path: Optional[str] = None,
):
    requests = load_trace(trace_path, max_requests)
    print(f"Loaded {len(requests)} requests from {trace_path}")

    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.monotonic()

        for i, req in enumerate(requests):
            # Calculate delay from trace timestamps
            if i > 0:
                delta_ms = req.timestamp - requests[i - 1].timestamp
                delay = max(0, delta_ms * rate_scale / 1000.0)
                await asyncio.sleep(delay)

            task = asyncio.create_task(
                send_request(session, endpoint, model, req, max_input_tokens, results, semaphore)
            )
            tasks.append(task)

            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - start_time
                ok = sum(1 for r in results if r.get("status") == "ok")
                print(f"  Dispatched {i+1}/{len(requests)}, completed {ok}, elapsed {elapsed:.1f}s")

        await asyncio.gather(*tasks)

    wall_time = time.monotonic() - start_time
    ok_results = [r for r in results if r.get("status") == "ok"]

    print(f"\n{'='*60}")
    print(f"Replay complete: {len(ok_results)}/{len(requests)} succeeded in {wall_time:.1f}s")
    if ok_results:
        ttfts = [r["ttft"] for r in ok_results if r["ttft"] is not None]
        total_times = [r["total_time"] for r in ok_results]
        output_tokens = sum(r["output_tokens"] for r in ok_results)

        ttfts.sort()
        total_times.sort()
        p = lambda lst, pct: lst[int(len(lst) * pct / 100)] if lst else 0

        print(f"Throughput: {output_tokens / wall_time:.1f} tok/s")
        print(f"TTFT  - P50: {p(ttfts, 50)*1000:.1f}ms  P90: {p(ttfts, 90)*1000:.1f}ms  P99: {p(ttfts, 99)*1000:.1f}ms")
        print(f"E2E   - P50: {p(total_times, 50):.2f}s  P90: {p(total_times, 90):.2f}s  P99: {p(total_times, 99):.2f}s")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mooncake trace replay for llm-d")
    parser.add_argument("--trace", required=True, help="Path to Mooncake JSONL trace file")
    parser.add_argument("--endpoint", default="http://localhost:8080", help="Gateway endpoint")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--rate-scale", type=float, default=0.001,
                        help="Scale factor for inter-arrival times (lower=faster replay)")
    parser.add_argument("--max-requests", type=int, default=None, help="Max requests to replay")
    parser.add_argument("--max-concurrency", type=int, default=64, help="Max concurrent requests")
    parser.add_argument("--max-input-tokens", type=int, default=8192, help="Max input tokens per request")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    asyncio.run(replay(
        trace_path=args.trace,
        endpoint=args.endpoint,
        model=args.model,
        rate_scale=args.rate_scale,
        max_requests=args.max_requests,
        max_concurrency=args.max_concurrency,
        max_input_tokens=args.max_input_tokens,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
