# mojo-mlxtend

`mojo-mlxtend` is a standalone Mojo port of the compute-heavy frequent-pattern
subset of [mlxtend](https://github.com/rasbt/mlxtend). It keeps the familiar
pandas-facing function names and signatures while moving support counting,
bitset intersections, population counts, and association-rule metrics into one
compiled Mojo shared library.

The parity target is mlxtend 0.25.0.

## Coverage

The public module `mojo_mlxtend.frequent_patterns` provides:

- `apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False, n_jobs=1)`
- `fpgrowth(df, min_support=0.5, null_values=False, use_colnames=False, max_len=None, verbose=0)`
- `fpmax(df, min_support=0.5, null_values=False, use_colnames=False, max_len=None, verbose=0)`
- `association_rules(df, num_itemsets=1, df_orig=None, null_values=False, metric="confidence", min_threshold=0.8, support_only=False, return_metrics=...)`

Dense boolean and 0/1 pandas DataFrames are supported. Pandas sparse DataFrames
are accepted and densified before crossing the FFI. `association_rules` computes
all twelve metrics returned by current mlxtend, including representativity,
Zhang's metric, Jaccard, certainty, and Kulczynski.

Not covered:

- `null_values=True`, whose missing-value-aware support denominators have
  different semantics rather than a different storage format
- storage-preserving sparse mining
- mlxtend modules outside frequent itemsets and association rules
- private helpers such as `generate_new_combinations`

Passing `null_values=True` raises `NotImplementedError` instead of silently
returning ordinary support values.

## Install

Install the pinned Mojo nightly and Python dependencies, then build the shared
library:

```bash
pixi install
pixi run build
```

Run the parity suite with:

```bash
pixi run test
```

## Usage

```python
import pandas as pd

from mojo_mlxtend.frequent_patterns import association_rules, fpgrowth

baskets = pd.DataFrame(
    [
        [True, True, False],
        [True, True, True],
        [True, False, True],
        [False, True, True],
    ],
    columns=["bread", "milk", "eggs"],
)

itemsets = fpgrowth(baskets, min_support=0.5, use_colnames=True)
rules = association_rules(
    itemsets,
    metric="confidence",
    min_threshold=0.5,
)

print(itemsets)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

The repository's Pixi activation sets `PYTHONPATH=python`, so the example runs
after `pixi install` and `pixi run build`.

## How it works

`apriori` performs the join and prune steps in Python, then sends batches of
candidates to Mojo. The kernel scans the zero-copy row-major `uint8` transaction
matrix with strided SIMD loads and a scalar remainder loop, without constructing
mlxtend's temporary three-dimensional boolean arrays. Independent candidates run
in parallel only above a work threshold; smaller calls stay serial.
`low_memory=True` uses smaller candidate batches.

The compatible `fpgrowth` entry point uses a vertical Eclat-style engine:
transactions are packed into item-major `uint64` bitsets, and Mojo computes
batched intersections and SIMD population counts. This differs internally from
mlxtend's FP-growth tree but returns the same frequent itemsets and supports.
Unbounded `fpmax` filters maximal sets from this SIMD bitset engine. Bounded
`fpmax` retains conditional FP trees because mlxtend's `max_len` pruning has
distinct edge-case behavior.

For association rules, Python expands each frequent itemset into antecedent and
consequent pairs directly into exact-size contiguous buffers. One Mojo call
computes all twelve metrics over the `float64` buffer, and the selection mask is
applied to the metric matrix once.

The Python process owns every allocation. C-contiguous NumPy buffer addresses
cross ctypes as 64-bit integers and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported C ABI. The single
compilation unit builds to `dist/libmojo-mlxtend.so`; no Python objects cross
the ABI.

There is intentionally no GPU path. Candidate counting, bitset intersection and
population counts, and rule metrics all remain below roughly 2 operations per
byte moved. They are bandwidth-bound and do not justify host/device copies or a
GPU implementation that loses to the CPU.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
72 logical CPUs, Python 3.13.14, and mlxtend 0.25.0. Times are the best of three
warm runs on identical DataFrames. Speedup is mlxtend time divided by
mojo-mlxtend time.

| case | mojo-mlxtend | mlxtend | speedup | result |
|---|---:|---:|---:|---:|
| apriori 120k x 24, max_len=3 | 87.55 ms | 1042.50 ms | 11.91x | 324 |
| fpgrowth 40k x 32 | 26.83 ms | 790.15 ms | 29.45x | 568 |
| fpmax 30k x 28 | 23.14 ms | 477.87 ms | 20.65x | 343 |
| association_rules 3,767 itemsets | 132.50 ms | 197.28 ms | 1.49x | 43,089 |

Benchmark results depend on transaction density, support threshold, item count,
and CPU. The benchmark script asserts that both implementations return the
same itemsets and supports, or the same rule pairs and metric values.
