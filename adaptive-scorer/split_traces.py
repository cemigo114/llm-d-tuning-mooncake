#!/usr/bin/env python3
"""
Split Mooncake traces into train/validation/test sets (60/20/20).

Splits are stratified by session to preserve multi-turn conversation structure.
Requests from the same session (consecutive hash_id overlap) stay together.

Usage:
  python split_traces.py \
    --input ../Mooncake/FAST25-release/traces/conversation_trace.jsonl \
    --output-dir ./data/splits \
    --seed 42
"""

import argparse
import json
import os
import random
from collections import defaultdict


def detect_sessions(trace: list[dict], overlap_threshold: float = 0.3) -> list[list[int]]:
    """Group requests into sessions based on hash_id overlap.
    Two consecutive requests are in the same session if they share
    >overlap_threshold fraction of hash_ids."""
    sessions = []
    current_session = [0]

    for i in range(1, len(trace)):
        prev_hashes = set(trace[i - 1]["hash_ids"])
        curr_hashes = set(trace[i]["hash_ids"])
        if not prev_hashes or not curr_hashes:
            current_session.append(i)
            continue
        overlap = len(prev_hashes & curr_hashes) / min(len(prev_hashes), len(curr_hashes))
        if overlap > overlap_threshold:
            current_session.append(i)
        else:
            sessions.append(current_session)
            current_session = [i]

    if current_session:
        sessions.append(current_session)

    return sessions


def split_sessions(
    sessions: list[list[int]],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Split sessions into train/val/test, preserving session boundaries."""
    rng = random.Random(seed)
    indices = list(range(len(sessions)))
    rng.shuffle(indices)

    n = len(indices)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_sessions = [sessions[i] for i in indices[:train_end]]
    val_sessions = [sessions[i] for i in indices[train_end:val_end]]
    test_sessions = [sessions[i] for i in indices[val_end:]]

    return train_sessions, val_sessions, test_sessions


def sessions_to_trace(trace: list[dict], sessions: list[list[int]]) -> list[dict]:
    """Flatten session groups back to a trace list, sorted by original index."""
    indices = sorted([i for session in sessions for i in session])
    return [trace[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Split Mooncake traces")
    parser.add_argument("--input", required=True, help="Input JSONL trace file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.input) as f:
        trace = [json.loads(line) for line in f]

    print(f"Total requests: {len(trace)}")

    sessions = detect_sessions(trace)
    print(f"Detected {len(sessions)} sessions (avg {len(trace)/len(sessions):.1f} requests/session)")

    train_s, val_s, test_s = split_sessions(
        sessions, args.train_frac, args.val_frac, args.seed
    )

    train = sessions_to_trace(trace, train_s)
    val = sessions_to_trace(trace, val_s)
    test = sessions_to_trace(trace, test_s)

    print(f"Train: {len(train)} requests ({len(train_s)} sessions)")
    print(f"Val:   {len(val)} requests ({len(val_s)} sessions)")
    print(f"Test:  {len(test)} requests ({len(test_s)} sessions)")

    os.makedirs(args.output_dir, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
