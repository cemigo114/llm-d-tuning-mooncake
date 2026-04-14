#!/usr/bin/env python3
"""
Analyze Mooncake trace replay results and produce comparison tables.

Usage:
  python analyze_results.py ../results/
  python analyze_results.py ../results/ --format markdown
  python analyze_results.py ../results/ --format csv
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def percentile(sorted_list: list[float], pct: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * pct / 100)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def analyze(data: list[dict]) -> dict:
    ok = [r for r in data if r.get("status") == "ok"]
    errors = [r for r in data if r.get("status") == "error"]

    if not ok:
        return {"total": len(data), "succeeded": 0, "failed": len(errors)}

    ttfts = sorted([r["ttft"] for r in ok if r.get("ttft") is not None])
    e2es = sorted([r["total_time"] for r in ok])
    output_tokens = sum(r.get("output_tokens", 0) for r in ok)
    input_lengths = [r.get("input_length", 0) for r in ok]

    return {
        "total": len(data),
        "succeeded": len(ok),
        "failed": len(errors),
        "output_tokens": output_tokens,
        "avg_input_length": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000,
        "ttft_p90_ms": percentile(ttfts, 90) * 1000,
        "ttft_p99_ms": percentile(ttfts, 99) * 1000,
        "e2e_p50_s": percentile(e2es, 50),
        "e2e_p90_s": percentile(e2es, 90),
        "e2e_p99_s": percentile(e2es, 99),
    }


def format_markdown(results: dict[str, dict]) -> str:
    lines = [
        "| Metric | " + " | ".join(results.keys()) + " |",
        "| --- | " + " | ".join(["---"] * len(results)) + " |",
    ]

    metrics = [
        ("Requests", lambda r: f"{r['succeeded']}/{r['total']}"),
        ("Output Tokens", lambda r: f"{r.get('output_tokens', 'N/A'):,}"),
        ("TTFT P50", lambda r: f"{r.get('ttft_p50_ms', 0):.1f} ms"),
        ("TTFT P90", lambda r: f"{r.get('ttft_p90_ms', 0):.1f} ms"),
        ("TTFT P99", lambda r: f"{r.get('ttft_p99_ms', 0):.1f} ms"),
        ("E2E P50", lambda r: f"{r.get('e2e_p50_s', 0):.2f} s"),
        ("E2E P90", lambda r: f"{r.get('e2e_p90_s', 0):.2f} s"),
        ("E2E P99", lambda r: f"{r.get('e2e_p99_s', 0):.2f} s"),
    ]

    for name, fmt in metrics:
        row = f"| {name} | " + " | ".join(
            fmt(r) if r.get("succeeded", 0) > 0 else "N/A"
            for r in results.values()
        ) + " |"
        lines.append(row)

    return "\n".join(lines)


def format_csv(results: dict[str, dict]) -> str:
    headers = ["metric"] + list(results.keys())
    lines = [",".join(headers)]

    metrics = [
        ("requests", lambda r: f"{r['succeeded']}/{r['total']}"),
        ("output_tokens", lambda r: str(r.get("output_tokens", ""))),
        ("ttft_p50_ms", lambda r: f"{r.get('ttft_p50_ms', 0):.1f}"),
        ("ttft_p90_ms", lambda r: f"{r.get('ttft_p90_ms', 0):.1f}"),
        ("ttft_p99_ms", lambda r: f"{r.get('ttft_p99_ms', 0):.1f}"),
        ("e2e_p50_s", lambda r: f"{r.get('e2e_p50_s', 0):.2f}"),
        ("e2e_p90_s", lambda r: f"{r.get('e2e_p90_s', 0):.2f}"),
        ("e2e_p99_s", lambda r: f"{r.get('e2e_p99_s', 0):.2f}"),
    ]

    for name, fmt in metrics:
        row = [name] + [
            fmt(r) if r.get("succeeded", 0) > 0 else ""
            for r in results.values()
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Mooncake replay results")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    parser.add_argument("--format", choices=["markdown", "csv", "text"], default="text")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Auto-discover and label result files
    label_map = {
        "mooncake_baseline_100": "Baseline (100)",
        "mooncake_conversation_500": "Co-located Conv",
        "mooncake_toolagent_500": "Co-located Tool",
        "pd_conversation_500": "P/D Conv",
        "pd_toolagent_500": "P/D Tool",
    }

    results = {}
    for json_file in sorted(results_dir.glob("*.json")):
        stem = json_file.stem
        label = label_map.get(stem, stem)
        data = load_results(str(json_file))
        results[label] = analyze(data)

    if not results:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    if args.format == "markdown":
        print(format_markdown(results))
    elif args.format == "csv":
        print(format_csv(results))
    else:
        for label, metrics in results.items():
            print(f"\n{'='*50}")
            print(f"  {label}")
            print(f"{'='*50}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
