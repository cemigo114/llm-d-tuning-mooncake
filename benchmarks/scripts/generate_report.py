#!/usr/bin/env python3
"""
Generate the llm-d playbook stage-by-stage gains report.

Reads JSON results from each stage and produces a markdown table showing
cumulative gains: throughput, TTFT, ITL, TPSU (1/ITL), and E2E.

Usage:
  python generate_report.py results/stage*.json
  python generate_report.py results/ --glob "stage_*.json"
"""

import argparse
import json
import os
import sys
from pathlib import Path


def pct(sorted_list: list[float], p: float) -> float:
    if not sorted_list:
        return 0.0
    idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
    return sorted_list[idx]


def analyze_stage(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    stage = data.get("stage", Path(path).stem)
    wall_time = data.get("wall_time_s", 0)
    requests = data.get("requests", data if isinstance(data, list) else [])

    # Handle both old format (list) and new format (dict with .requests)
    if isinstance(requests, dict):
        requests = requests.get("requests", [])

    ok = [r for r in requests if r.get("status") == "ok"]
    ttfts = sorted([r["ttft"] for r in ok if r.get("ttft") is not None])
    itls = sorted([r["itl"] for r in ok if r.get("itl") is not None])
    e2es = sorted([r["total_time"] for r in ok])
    total_tokens = sum(r.get("output_tokens", 0) for r in ok)
    throughput = total_tokens / wall_time if wall_time > 0 else 0

    itl_p50 = pct(itls, 50) if itls else None
    tpsu_p50 = 1.0 / itl_p50 if itl_p50 and itl_p50 > 0 else None

    return {
        "stage": stage,
        "file": Path(path).name,
        "requests": f"{len(ok)}/{len(requests)}",
        "wall_time": wall_time,
        "throughput": throughput,
        "ttft_p50": pct(ttfts, 50) * 1000 if ttfts else None,
        "ttft_p90": pct(ttfts, 90) * 1000 if ttfts else None,
        "ttft_p99": pct(ttfts, 99) * 1000 if ttfts else None,
        "itl_p50": itl_p50 * 1000 if itl_p50 else None,
        "itl_p90": pct(itls, 90) * 1000 if itls else None,
        "tpsu_p50": tpsu_p50,
        "e2e_p50": pct(e2es, 50) if e2es else None,
        "e2e_p90": pct(e2es, 90) if e2es else None,
    }


def delta_str(current, baseline):
    if current is None or baseline is None or baseline == 0:
        return ""
    d = (current - baseline) / baseline * 100
    return f"{d:+.0f}%"


def fmt(val, unit=""):
    if val is None:
        return "—"
    if unit == "ms":
        return f"{val:.0f}"
    if unit == "s":
        return f"{val:.2f}"
    if unit == "tok/s":
        return f"{val:,.0f}"
    if unit == "tpsu":
        return f"{val:.1f}"
    return f"{val}"


def generate_report(stages: list[dict]) -> str:
    if not stages:
        return "No stage data found."

    baseline = stages[0]
    lines = []

    lines.append("# llm-d Playbook: Stage-by-Stage Performance Gains")
    lines.append("")
    lines.append("Each stage adds optimization on top of the previous, measured on")
    lines.append("the same cluster (2x8 H200, 16 GPUs) with Mooncake conversation trace.")
    lines.append("")

    # Main table
    header = "| # | Stage | Throughput (tok/s) | TTFT P50 (ms) | TTFT P90 (ms) | ITL P50 (ms) | TPSU P50 (tok/s/user) | E2E P50 (s) |"
    sep =    "| - | ----- | -----------------: | ------------: | ------------: | -----------: | --------------------: | ----------: |"
    lines.append(header)
    lines.append(sep)

    for i, s in enumerate(stages):
        num = i + 1
        throughput_d = delta_str(s["throughput"], baseline["throughput"]) if i > 0 else "baseline"
        ttft_d = delta_str(s["ttft_p50"], baseline["ttft_p50"]) if i > 0 else ""
        itl_d = delta_str(s["itl_p50"], baseline["itl_p50"]) if i > 0 and s["itl_p50"] and baseline["itl_p50"] else ""

        row = (
            f"| {num} | **{s['stage']}** "
            f"| {fmt(s['throughput'], 'tok/s')} ({throughput_d}) "
            f"| {fmt(s['ttft_p50'], 'ms')} {('('+ttft_d+')') if ttft_d else ''} "
            f"| {fmt(s['ttft_p90'], 'ms')} "
            f"| {fmt(s['itl_p50'], 'ms')} {('('+itl_d+')') if itl_d else ''} "
            f"| {fmt(s['tpsu_p50'], 'tpsu')} "
            f"| {fmt(s['e2e_p50'], 's')} |"
        )
        lines.append(row)

    # Cumulative gains summary
    last = stages[-1]
    lines.append("")
    lines.append("### Cumulative Gains (Stage 1 → Stage 5)")
    lines.append("")

    metrics = [
        ("Throughput", baseline["throughput"], last["throughput"], "tok/s"),
        ("TTFT P50", baseline["ttft_p50"], last["ttft_p50"], "ms"),
        ("TTFT P90", baseline["ttft_p90"], last["ttft_p90"], "ms"),
        ("ITL P50", baseline.get("itl_p50"), last.get("itl_p50"), "ms"),
        ("TPSU P50", baseline.get("tpsu_p50"), last.get("tpsu_p50"), "tok/s/user"),
        ("E2E P50", baseline["e2e_p50"], last["e2e_p50"], "s"),
    ]

    lines.append("| Metric | Stage 1 (Baseline) | Stage 5 (Final) | Improvement |")
    lines.append("| ------ | -----------------: | --------------: | ----------: |")
    for name, base_val, final_val, unit in metrics:
        if base_val is not None and final_val is not None and base_val != 0:
            delta = (final_val - base_val) / base_val * 100
            lines.append(
                f"| {name} | {fmt(base_val, unit)} {unit} | {fmt(final_val, unit)} {unit} | {delta:+.0f}% |"
            )

    # Stage descriptions
    lines.append("")
    lines.append("### What Each Stage Adds")
    lines.append("")
    for i, s in enumerate(stages):
        lines.append(f"**Stage {i+1}: {s['stage']}**")
        lines.append(f"- Requests: {s['requests']}, Wall time: {s['wall_time']:.1f}s")
        lines.append(f"- File: `{s['file']}`")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Stage result JSON files in order")
    args = parser.parse_args()

    stages = []
    for path in sorted(args.files):
        if os.path.isfile(path) and path.endswith(".json"):
            stages.append(analyze_stage(path))
        elif os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.startswith("stage") and f.endswith(".json"):
                    stages.append(analyze_stage(os.path.join(path, f)))

    print(generate_report(stages))


if __name__ == "__main__":
    main()
