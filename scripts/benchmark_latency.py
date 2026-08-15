"""Runs N real queries through the pipeline (text path, so STT jitter and
network variance don't pollute the retrieval numbers) and reports P50/P70/P100
latency for each stage plus end-to-end. Retrieval-stage numbers are what we
hold to the <200ms target; full-pipeline numbers (including the LLM call)
are reported honestly alongside them rather than hidden.

Usage:
    python scripts/benchmark_latency.py --n 100 --out latency_report.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.pipeline import run_pipeline  # noqa: E402

# Swap this for a sample pulled from the actual MSMARCO-XI query column —
# kept as a placeholder list so this script runs standalone for a smoke test.
SAMPLE_QUERIES = [
    "what is the capital of india",
    "how does photosynthesis work",
    "when was the taj mahal built",
    "symptoms of dehydration",
    "how to apply for a passport in india",
    # ... extend with real queries sampled from the dataset at benchmark time
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


async def run_benchmark(n: int, out_path: str) -> None:
    queries = (SAMPLE_QUERIES * ((n // len(SAMPLE_QUERIES)) + 1))[:n]

    rows = []
    stage_totals: dict[str, list[float]] = {}
    end_to_end: list[float] = []

    for q in queries:
        response = await run_pipeline(text_query=q)
        end_to_end.append(response.total_latency_ms)
        rows.append(
            {"query": q, "total_ms": response.total_latency_ms, "refused": response.refused,
             **{f"{k}_ms": v for k, v in response.stage_latencies_ms.items()}}
        )
        for stage, ms in response.stage_latencies_ms.items():
            stage_totals.setdefault(stage, []).append(ms)
        time.sleep(2.5)  # stay under Groq's free-tier requests-per-minute limit

    with open(out_path, "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} query results to {out_path}\n")
    print(f"{'stage':<25}{'P50 (ms)':>12}{'P70 (ms)':>12}{'P100 (ms)':>12}")
    retrieval_stages = {"input_guardrail", "retrieval", "grounding_guardrail"}
    retrieval_total = [
        sum(response_stage_vals) for response_stage_vals in zip(
            *(stage_totals[s] for s in retrieval_stages if s in stage_totals)
        )
    ] if retrieval_stages & stage_totals.keys() else []

    for stage, values in stage_totals.items():
        print(f"{stage:<25}{percentile(values,50):>12.1f}{percentile(values,70):>12.1f}{percentile(values,100):>12.1f}")

    if retrieval_total:
        print(f"{'RETRIEVAL (target <200ms)':<25}{percentile(retrieval_total,50):>12.1f}"
              f"{percentile(retrieval_total,70):>12.1f}{percentile(retrieval_total,100):>12.1f}")
    print(f"{'FULL PIPELINE (honest)':<25}{percentile(end_to_end,50):>12.1f}"
          f"{percentile(end_to_end,70):>12.1f}{percentile(end_to_end,100):>12.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="latency_report.csv")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.n, args.out))
