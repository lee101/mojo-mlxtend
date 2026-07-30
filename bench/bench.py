"""Benchmarks against mlxtend 0.25 on identical transaction data."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mlxtend.frequent_patterns import (  # noqa: E402
    apriori as upstream_apriori,
    association_rules as upstream_association_rules,
    fpgrowth as upstream_fpgrowth,
    fpmax as upstream_fpmax,
)

from mojo_mlxtend.frequent_patterns import (  # noqa: E402
    apriori,
    association_rules,
    fpgrowth,
    fpmax,
)


def timeit(function, repeat=3):
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def transactions(rows: int, columns: int, probability: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.random((rows, max(1, (columns + 3) // 4)))
    values = rng.random((rows, columns)) < probability
    for column in range(columns):
        values[:, column] |= latent[:, column // 4] < probability * 0.45
    return pd.DataFrame(values, columns=[f"item_{index}" for index in range(columns)])


def machine() -> str:
    model = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {os.cpu_count()} logical CPUs; Python {platform.python_version()}"


def main():
    dense = transactions(120_000, 24, 0.12, 1)
    medium = transactions(40_000, 32, 0.09, 2)
    maximal_data = transactions(30_000, 28, 0.10, 3)

    cases = [
        (
            "apriori 120k x 24, max_len=3",
            lambda: apriori(dense, min_support=0.018, max_len=3, n_jobs=-1),
            lambda: upstream_apriori(
                dense, min_support=0.018, max_len=3, n_jobs=-1
            ),
        ),
        (
            "fpgrowth 40k x 32",
            lambda: fpgrowth(medium, min_support=0.012, max_len=4),
            lambda: upstream_fpgrowth(medium, min_support=0.012, max_len=4),
        ),
        (
            "fpmax 30k x 28",
            lambda: fpmax(maximal_data, min_support=0.014),
            lambda: upstream_fpmax(maximal_data, min_support=0.014),
        ),
    ]

    frequent = upstream_fpgrowth(
        transactions(20_000, 18, 0.18, 4),
        min_support=0.006,
        max_len=5,
    )
    cases.append(
        (
            f"association_rules {len(frequent):,} itemsets",
            lambda: association_rules(
                frequent, metric="confidence", min_threshold=0.05
            ),
            lambda: upstream_association_rules(
                frequent, metric="confidence", min_threshold=0.05
            ),
        )
    )

    print(f"Machine: {machine()}; mlxtend {version('mlxtend')}")
    print()
    print("| case | mojo-mlxtend | mlxtend | speedup | result |")
    print("|---|---:|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_time, mojo_result = timeit(ours)
        upstream_time, upstream_result = timeit(theirs)
        result_count = len(mojo_result)
        assert result_count == len(upstream_result)
        if "association_rules" in name:
            mojo_keys = list(
                zip(mojo_result["antecedents"], mojo_result["consequents"])
            )
            upstream_positions = {
                key: position
                for position, key in enumerate(
                    zip(
                        upstream_result["antecedents"],
                        upstream_result["consequents"],
                    )
                )
            }
            assert set(mojo_keys) == upstream_positions.keys()
            ordered_upstream = upstream_result.iloc[
                [upstream_positions[key] for key in mojo_keys], 2:
            ]
            assert np.allclose(
                mojo_result.iloc[:, 2:].to_numpy(dtype=float),
                ordered_upstream.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-13,
            )
        else:
            mojo_itemsets = dict(zip(mojo_result["itemsets"], mojo_result["support"]))
            upstream_itemsets = dict(
                zip(upstream_result["itemsets"], upstream_result["support"])
            )
            assert mojo_itemsets.keys() == upstream_itemsets.keys()
            assert np.allclose(
                list(mojo_itemsets.values()),
                [upstream_itemsets[itemset] for itemset in mojo_itemsets],
                rtol=1e-13,
            )
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{upstream_time * 1000:.2f} ms | {upstream_time / mojo_time:.2f}x | "
            f"{result_count:,} |"
        )


if __name__ == "__main__":
    main()
