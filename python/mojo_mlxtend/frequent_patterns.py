"""Drop-in frequent-pattern functions accelerated by Mojo."""

from __future__ import annotations

import itertools
import math
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from ._lib import addr, check_status, lib

_METRICS = [
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "representativity",
    "leverage",
    "conviction",
    "zhangs_metric",
    "jaccard",
    "certainty",
    "kulczynski",
]
_METRIC_INDEX = {name: index for index, name in enumerate(_METRICS)}


def _threads(n_jobs: int | None = None) -> int:
    if n_jobs == 1:
        return 1
    available = os.cpu_count() or 1
    if n_jobs is None or n_jobs < 0:
        return min(available, 16)
    return max(1, min(int(n_jobs), available))


def _validate_frame(df, null_values: bool = False) -> np.ndarray:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if null_values:
        raise NotImplementedError("null_values=True is not covered by mojo-mlxtend")
    if df.size == 0:
        return np.ascontiguousarray(df.to_numpy(dtype=np.uint8))
    if all(isinstance(dtype, pd.SparseDtype) for dtype in df.dtypes):
        values = df.sparse.to_dense().to_numpy()
    else:
        values = df.to_numpy()
    if pd.isna(values).any():
        raise ValueError("NaN values are not permitted in the DataFrame when null_values=False.")
    valid = (values == 0) | (values == 1)
    if not bool(np.all(valid)):
        bad = values[~valid][0]
        raise ValueError(
            "The allowed values for a DataFrame are True, False, 0, 1. "
            f"Found value {bad}"
        )
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in df.dtypes):
        warnings.warn(
            "DataFrames with non-bool types result in worse computational performance. "
            "Please use a DataFrame with bool type",
            DeprecationWarning,
            stacklevel=2,
        )
    return np.ascontiguousarray(values, dtype=np.uint8)


def _minimum_count(min_support: float, rows: int) -> int:
    if min_support <= 0.0 or min_support > 1.0:
        raise ValueError(
            "`min_support` must be a positive number within the interval `(0, 1]`. "
            f"Got {min_support}."
        )
    return math.ceil(min_support * rows)


def _candidate_generator(previous: np.ndarray):
    combinations = sorted(tuple(int(item) for item in row) for row in previous)
    combination_set = set(combinations)
    for index, left in enumerate(combinations):
        prefix = left[:-1]
        for right in combinations[index + 1 :]:
            if prefix != right[:-1]:
                break
            candidate = left + (right[-1],)
            if all(
                candidate[:drop] + candidate[drop + 1 :] in combination_set
                for drop in range(len(candidate) - 2)
            ):
                yield candidate


def _count_candidates(
    values: np.ndarray,
    candidates: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    candidates = np.ascontiguousarray(candidates, dtype=np.int64)
    counts = np.empty(candidates.shape[0], dtype=np.int64)
    if candidates.shape[0]:
        if values.dtype != np.uint8 or values.ndim != 2 or not values.flags.c_contiguous:
            raise TypeError("candidate data must be a C-contiguous uint8 matrix")
        if candidates.ndim != 2 or candidates.shape[1] == 0:
            raise ValueError("candidates must be a non-empty-width matrix")
        check_status(
            "mmlx_count_candidates",
            lib().mmlx_count_candidates(
                addr(values),
                addr(candidates),
                addr(counts),
                values.shape[0],
                values.shape[1],
                candidates.shape[0],
                candidates.shape[1],
                _threads(n_jobs),
            ),
        )
    return counts


def _mine_apriori(
    values: np.ndarray,
    min_count: int,
    max_len: int | None,
    low_memory: bool,
    verbose: int,
    n_jobs: int,
) -> tuple[list[tuple[int, ...]], list[int]]:
    rows, columns = values.shape
    if rows == 0 or columns == 0:
        return [], []
    singleton_candidates = np.arange(columns, dtype=np.int64).reshape(-1, 1)
    singleton_counts = _count_candidates(values, singleton_candidates, n_jobs)
    keep = singleton_counts >= min_count
    previous = singleton_candidates[keep]
    itemsets = [tuple(row) for row in previous.tolist()]
    supports = singleton_counts[keep].tolist()
    width = 1
    limit = max_len if max_len else columns
    batch_size = 1_024 if low_memory else 65_536
    while len(previous) and width < limit:
        width += 1
        frequent_candidates: list[np.ndarray] = []
        frequent_counts: list[np.ndarray] = []
        generated = 0
        generator = _candidate_generator(previous)
        while True:
            batch = list(itertools.islice(generator, batch_size))
            if not batch:
                break
            generated += len(batch)
            candidate_array = np.asarray(batch, dtype=np.int64)
            counts = _count_candidates(values, candidate_array, n_jobs)
            mask = counts >= min_count
            if np.any(mask):
                frequent_candidates.append(candidate_array[mask])
                frequent_counts.append(counts[mask])
        if verbose:
            print(
                f"\rProcessing {generated * width} combinations | "
                f"Sampling itemset size {width}",
                end="",
            )
        if not frequent_candidates:
            break
        previous = np.concatenate(frequent_candidates)
        level_counts = np.concatenate(frequent_counts)
        itemsets.extend(tuple(row) for row in previous.tolist())
        supports.extend(int(count) for count in level_counts)
    if verbose:
        print()
    return itemsets, supports


def _vertical_bits(values: np.ndarray) -> np.ndarray:
    packed = np.packbits(values.T, axis=1, bitorder="little")
    padded_bytes = ((packed.shape[1] + 7) // 8) * 8
    if padded_bytes != packed.shape[1]:
        padded = np.zeros((packed.shape[0], padded_bytes), dtype=np.uint8)
        padded[:, : packed.shape[1]] = packed
        packed = padded
    return np.ascontiguousarray(packed).view(np.uint64)


def _intersect_items(
    vertical: np.ndarray,
    item_ids: np.ndarray,
    parent: np.ndarray | None,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    item_ids = np.ascontiguousarray(item_ids, dtype=np.int64)
    children = np.empty((item_ids.size, vertical.shape[1]), dtype=np.uint64)
    counts = np.empty(item_ids.size, dtype=np.int64)
    if item_ids.size:
        if vertical.dtype != np.uint64 or vertical.ndim != 2 or not vertical.flags.c_contiguous:
            raise TypeError("vertical data must be a C-contiguous uint64 matrix")
        if parent is not None and (
            parent.dtype != np.uint64
            or parent.ndim != 1
            or not parent.flags.c_contiguous
            or parent.shape[0] != vertical.shape[1]
        ):
            raise TypeError("parent must be one contiguous uint64 bitset")
        check_status(
            "mmlx_intersect_count",
            lib().mmlx_intersect_count(
                0 if parent is None else addr(parent),
                addr(vertical),
                addr(item_ids),
                addr(children),
                addr(counts),
                item_ids.size,
                vertical.shape[0],
                vertical.shape[1],
                threads,
            ),
        )
    return children, counts


def _mine_vertical(
    values: np.ndarray,
    min_count: int,
    max_len: int | None,
    verbose: int,
) -> tuple[list[tuple[int, ...]], list[int]]:
    rows, columns = values.shape
    if rows == 0 or columns == 0:
        return [], []
    vertical = _vertical_bits(values)
    limit = max_len if max_len else columns
    threads = _threads(-1)
    itemsets: list[tuple[int, ...]] = []
    supports: list[int] = []
    visited = 0

    def extend(prefix: tuple[int, ...], parent: np.ndarray | None, candidates: np.ndarray):
        nonlocal visited
        if not len(candidates) or len(prefix) >= limit:
            return
        children, counts = _intersect_items(vertical, candidates, parent, threads)
        valid_positions = np.flatnonzero(counts >= min_count)
        for position in valid_positions:
            item = int(candidates[position])
            child_prefix = prefix + (item,)
            itemsets.append(child_prefix)
            supports.append(int(counts[position]))
            visited += 1
            if len(child_prefix) < limit:
                extend(child_prefix, children[position], candidates[position + 1 :])

    extend((), None, np.arange(columns, dtype=np.int64))
    if verbose:
        print(f"\r{visited} itemsets mined", end="")
        print()
    return itemsets, supports


def _result_frame(
    itemsets: list[tuple[int, ...]],
    counts: list[int],
    rows: int,
    columns,
    use_colnames: bool,
) -> pd.DataFrame:
    if use_colnames:
        sets = [frozenset(columns[item] for item in itemset) for itemset in itemsets]
    else:
        sets = [frozenset(itemset) for itemset in itemsets]
    supports = np.asarray(counts, dtype=np.float64) / float(rows) if rows else []
    return pd.DataFrame({"support": supports, "itemsets": pd.Series(sets, dtype=object)})


def apriori(
    df,
    min_support=0.5,
    use_colnames=False,
    max_len=None,
    verbose=0,
    low_memory=False,
    n_jobs=1,
):
    values = _validate_frame(df)
    min_count = _minimum_count(min_support, values.shape[0])
    itemsets, counts = _mine_apriori(
        values, min_count, max_len, low_memory, verbose, n_jobs
    )
    return _result_frame(itemsets, counts, values.shape[0], df.columns, use_colnames)


def fpgrowth(
    df, min_support=0.5, null_values=False, use_colnames=False, max_len=None, verbose=0
):
    values = _validate_frame(df, null_values)
    min_count = _minimum_count(min_support, values.shape[0])
    itemsets, counts = _mine_vertical(values, min_count, max_len, verbose)
    return _result_frame(itemsets, counts, values.shape[0], df.columns, use_colnames)


class _FPNode:
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: dict[int, _FPNode] = {}

    def path_from_root(self) -> list[int]:
        path: list[int] = []
        node = self.parent
        while node is not None and node.item is not None:
            path.append(node.item)
            node = node.parent
        path.reverse()
        return path


class _FPTree:
    def __init__(self, rank: dict[int, int]):
        self.root = _FPNode()
        self.nodes: defaultdict[int, list[_FPNode]] = defaultdict(list)
        self.cond_items: list[int] = []
        self.rank = rank

    def insert(self, itemset, count=1):
        self.root.count += count
        node = self.root
        for item in itemset:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, count, node)
                node.children[item] = child
                self.nodes[item].append(child)
            else:
                child.count += count
            node = child

    def is_path(self) -> bool:
        node = self.root
        while node.children:
            if len(node.children) != 1:
                return False
            node = next(iter(node.children.values()))
        return True

    def conditional(self, item: int, minimum: int):
        branches: list[tuple[list[int], int]] = []
        counts: defaultdict[int, int] = defaultdict(int)
        for node in self.nodes[item]:
            branch = node.path_from_root()
            branches.append((branch, node.count))
            for branch_item in branch:
                counts[branch_item] += node.count
        items = [branch_item for branch_item, count in counts.items() if count >= minimum]
        items.sort(key=counts.get)
        tree = _FPTree({branch_item: rank for rank, branch_item in enumerate(items)})
        for branch, count in branches:
            kept = [branch_item for branch_item in branch if branch_item in tree.rank]
            kept.sort(key=tree.rank.get, reverse=True)
            tree.insert(kept, count)
        tree.cond_items = self.cond_items + [item]
        return tree


def _setup_fp_tree(values: np.ndarray, minimum: int) -> _FPTree:
    columns = values.shape[1]
    candidates = np.arange(columns, dtype=np.int64).reshape(-1, 1)
    counts = _count_candidates(values, candidates, -1)
    items = np.flatnonzero(counts >= minimum)
    ordered = items[np.argsort(counts[items])]
    rank = {int(item): index for index, item in enumerate(ordered)}
    tree = _FPTree(rank)
    for row in values:
        itemset = [int(item) for item in np.flatnonzero(row) if int(item) in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert(itemset)
    return tree


def _mine_fpmax(
    values: np.ndarray,
    minimum: int,
    max_len: int | None,
    verbose: int,
) -> tuple[list[tuple[int, ...]], list[int]]:
    if values.shape[0] == 0 or values.shape[1] == 0:
        return [], []
    if max_len is None:
        frequent_itemsets, frequent_counts = _mine_vertical(
            values, minimum, None, 0
        )
        maximal_sets: list[frozenset[int]] = []
        itemsets: list[tuple[int, ...]] = []
        supports: list[int] = []
        for index in sorted(
            range(len(frequent_itemsets)),
            key=lambda position: len(frequent_itemsets[position]),
            reverse=True,
        ):
            candidate = frozenset(frequent_itemsets[index])
            if any(candidate.issubset(maximal) for maximal in maximal_sets):
                continue
            maximal_sets.append(candidate)
            itemsets.append(frequent_itemsets[index])
            supports.append(frequent_counts[index])
        if verbose:
            print(f"\r{len(itemsets)} maximal itemsets mined")
        return itemsets, supports
    tree = _setup_fp_tree(values, minimum)
    global_rank = tree.rank.copy()
    maximal_sets: list[frozenset[int]] = []
    itemsets: list[tuple[int, ...]] = []
    supports: list[int] = []

    def contains(candidate) -> bool:
        candidate_set = frozenset(candidate)
        return any(candidate_set.issubset(itemset) for itemset in maximal_sets)

    def step(current: _FPTree):
        items = list(current.nodes)
        largest = sorted(current.cond_items + items, key=global_rank.get)
        if not largest:
            return
        if current.is_path():
            if not contains(largest):
                largest.reverse()
                maximal_sets.append(frozenset(largest))
                if max_len is None or len(largest) <= max_len:
                    support = current.root.count
                    if items:
                        support = min(current.nodes[item][0].count for item in items)
                    itemsets.append(tuple(largest))
                    supports.append(int(support))
        elif not max_len or max_len > len(current.cond_items):
            items.sort(key=current.rank.get)
            for item in items:
                if contains(largest):
                    return
                largest.remove(item)
                step(current.conditional(item, minimum))

    step(tree)
    if verbose:
        print(f"\r{len(itemsets)} maximal itemsets mined")
    return itemsets, supports


def fpmax(
    df, min_support=0.5, null_values=False, use_colnames=False, max_len=None, verbose=0
):
    values = _validate_frame(df, null_values)
    min_count = _minimum_count(min_support, values.shape[0])
    kept_itemsets, kept_counts = _mine_fpmax(values, min_count, max_len, verbose)
    return _result_frame(
        kept_itemsets, kept_counts, values.shape[0], df.columns, use_colnames
    )


def _rule_metric_matrix(source: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(source, dtype=np.float64)
    destination = np.empty((source.shape[0], len(_METRICS)), dtype=np.float64)
    if source.shape[0]:
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("rule metric source must have exactly three columns")
        check_status(
            "mmlx_rule_metrics",
            lib().mmlx_rule_metrics(
                addr(source), addr(destination), source.shape[0], _threads(-1)
            ),
        )
    return destination


def association_rules(
    df: pd.DataFrame,
    num_itemsets: int | None = 1,
    df_orig: pd.DataFrame | None = None,
    null_values=False,
    metric="confidence",
    min_threshold=0.8,
    support_only=False,
    return_metrics: list = _METRICS,
) -> pd.DataFrame:
    if null_values:
        raise NotImplementedError("null_values=True is not covered by mojo-mlxtend")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df_orig is not None:
        _validate_frame(df_orig)
    if not len(df):
        raise ValueError("The input DataFrame `df` containing the frequent itemsets is empty.")
    if not {"support", "itemsets"}.issubset(df.columns):
        raise ValueError("Dataframe needs to contain the columns 'support' and 'itemsets'")
    if support_only:
        metric = "support"
    elif metric not in _METRIC_INDEX:
        raise ValueError(f"Metric must be 'confidence' or 'lift', got '{metric}'")
    unknown_metrics = [name for name in return_metrics if name not in _METRIC_INDEX]
    if unknown_metrics:
        raise ValueError(f"Unknown return metric: {unknown_metrics[0]}")

    support_by_itemset = {
        frozenset(
            int(item) if isinstance(item, np.generic) else item for item in itemset
        ): float(support)
        for support, itemset in zip(df["support"], df["itemsets"])
    }
    rule_count = sum(
        (1 << len(itemset)) - 2
        for itemset in support_by_itemset
        if len(itemset) > 1
    )
    antecedents = np.empty(rule_count, dtype=object)
    consequents = np.empty(rule_count, dtype=object)
    source = np.empty((rule_count, 3), dtype=np.float64)
    position = 0
    for itemset, combined_support in support_by_itemset.items():
        for size in range(len(itemset) - 1, 0, -1):
            for combination in itertools.combinations(itemset, size):
                antecedent = frozenset(combination)
                consequent = itemset.difference(antecedent)
                if support_only:
                    antecedent_support = consequent_support = 1.0
                else:
                    try:
                        antecedent_support = support_by_itemset[antecedent]
                        consequent_support = support_by_itemset[consequent]
                    except KeyError as error:
                        raise KeyError(
                            f"{error}You are likely getting this error because the DataFrame "
                            "is missing antecedent and/or consequent information. You can try "
                            "using the `support_only=True` option"
                        ) from error
                antecedents[position] = antecedent
                consequents[position] = consequent
                source[position, 0] = combined_support
                source[position, 1] = antecedent_support
                source[position, 2] = consequent_support
                position += 1

    if not rule_count:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)
    metrics = _rule_metric_matrix(source)
    metric_values = source[:, 0] if support_only else metrics[:, _METRIC_INDEX[metric]]
    keep = metric_values >= min_threshold
    if not np.any(keep):
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

    selected_source = source[keep]
    if support_only:
        numeric = np.full((selected_source.shape[0], len(return_metrics)), np.nan)
        for index, name in enumerate(return_metrics):
            if name == "support":
                numeric[:, index] = selected_source[:, 0]
    else:
        selected_metrics = metrics[keep]
        if return_metrics == _METRICS:
            numeric = selected_metrics
        else:
            numeric = selected_metrics[
                :, [_METRIC_INDEX[name] for name in return_metrics]
            ]
    result = pd.DataFrame(numeric, columns=return_metrics)
    result.insert(0, "consequents", consequents[keep])
    result.insert(0, "antecedents", antecedents[keep])
    if support_only:
        result["support"] = selected_source[:, 0]
    return result
