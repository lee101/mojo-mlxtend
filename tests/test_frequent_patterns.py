from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import (
    apriori as upstream_apriori,
    association_rules as upstream_association_rules,
    fpgrowth as upstream_fpgrowth,
    fpmax as upstream_fpmax,
)

from mojo_mlxtend.frequent_patterns import (
    _count_candidates,
    _intersect_items,
    _rule_metric_matrix,
    apriori,
    association_rules,
    fpgrowth,
    fpmax,
)
from mojo_mlxtend._lib import lib


def basket() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        columns=["Apple", "Bananas", "Beer", "Chicken", "Milk", "Rice"],
        dtype=bool,
    )


def random_transactions(rows=350, columns=14, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probabilities = np.linspace(0.15, 0.65, columns)
    values = rng.random((rows, columns)) < probabilities
    return pd.DataFrame(values, columns=[f"item_{index}" for index in range(columns)])


def canonical(frame: pd.DataFrame) -> dict[frozenset, float]:
    return {
        frozenset(itemset): float(support)
        for support, itemset in zip(frame["support"], frame["itemsets"])
    }


def assert_itemset_parity(ours: pd.DataFrame, theirs: pd.DataFrame):
    left, right = canonical(ours), canonical(theirs)
    assert left.keys() == right.keys()
    for itemset in left:
        assert left[itemset] == pytest.approx(right[itemset], abs=1e-15)


@pytest.mark.parametrize("minimum", [0.25, 0.5, 0.75, 1.0])
def test_apriori_basket_parity(minimum):
    frame = basket()
    assert_itemset_parity(
        apriori(frame, min_support=minimum),
        upstream_apriori(frame, min_support=minimum),
    )


@pytest.mark.parametrize("max_len", [1, 2, 3, None])
def test_apriori_random_parity(max_len):
    frame = random_transactions()
    assert_itemset_parity(
        apriori(frame, min_support=0.12, max_len=max_len),
        upstream_apriori(frame, min_support=0.12, max_len=max_len),
    )


def test_apriori_colnames_low_memory_and_parallel():
    frame = random_transactions(rows=500, columns=16)
    ours = apriori(
        frame,
        min_support=0.1,
        use_colnames=True,
        max_len=3,
        low_memory=True,
        n_jobs=2,
    )
    theirs = upstream_apriori(
        frame,
        min_support=0.1,
        use_colnames=True,
        max_len=3,
        low_memory=True,
        n_jobs=2,
    )
    assert_itemset_parity(ours, theirs)


@pytest.mark.parametrize("algorithm,reference", [(fpgrowth, upstream_fpgrowth), (fpmax, upstream_fpmax)])
def test_vertical_miners_basket_parity(algorithm, reference):
    frame = basket()
    assert_itemset_parity(
        algorithm(frame, min_support=0.25, use_colnames=True),
        reference(frame, min_support=0.25, use_colnames=True),
    )


@pytest.mark.parametrize("max_len", [2, 3, None])
def test_fpgrowth_random_parity(max_len):
    frame = random_transactions(rows=450, columns=13)
    assert_itemset_parity(
        fpgrowth(frame, min_support=0.1, max_len=max_len),
        upstream_fpgrowth(frame, min_support=0.1, max_len=max_len),
    )


@pytest.mark.parametrize("max_len", [1, 2, 3, None])
def test_fpmax_random_parity(max_len):
    frame = random_transactions(rows=400, columns=12)
    assert_itemset_parity(
        fpmax(frame, min_support=0.1, max_len=max_len),
        upstream_fpmax(frame, min_support=0.1, max_len=max_len),
    )


def test_fpmax_unbounded_simd_tail_parity():
    frame = random_transactions(rows=321, columns=15, seed=19)
    assert_itemset_parity(
        fpmax(frame, min_support=0.11),
        upstream_fpmax(frame, min_support=0.11),
    )


def test_fpmax_unbounded_parallel_threshold_parity():
    rows = 16_384
    columns = 64
    values = np.zeros((rows, columns), dtype=bool)
    row_ids = np.arange(rows)
    values[row_ids, row_ids % columns] = True
    frame = pd.DataFrame(values)
    assert_itemset_parity(
        fpmax(frame, min_support=0.01),
        upstream_fpmax(frame, min_support=0.01),
    )


def test_sparse_dataframe_parity():
    frame = random_transactions(rows=300, columns=10)
    sparse = frame.astype(pd.SparseDtype(bool, False))
    assert_itemset_parity(
        apriori(sparse, min_support=0.15),
        upstream_apriori(sparse, min_support=0.15),
    )
    assert_itemset_parity(
        fpgrowth(sparse, min_support=0.15),
        upstream_fpgrowth(sparse, min_support=0.15),
    )


def test_empty_transaction_frame():
    frame = pd.DataFrame(columns=["a", "b"], dtype=bool)
    assert apriori(frame).empty
    assert fpgrowth(frame).empty
    assert fpmax(frame).empty


@pytest.mark.parametrize("algorithm", [apriori, fpgrowth, fpmax])
def test_invalid_support_matches_upstream_behavior(algorithm):
    with pytest.raises(ValueError, match="min_support"):
        algorithm(basket(), min_support=0)
    with pytest.raises(ValueError, match="min_support"):
        algorithm(basket(), min_support=1.01)


def test_invalid_values_and_nan_rejected():
    with pytest.raises(ValueError, match="allowed values"):
        apriori(pd.DataFrame({"a": [0, 2]}))
    with pytest.raises(ValueError, match="NaN"):
        fpgrowth(pd.DataFrame({"a": [True, np.nan]}))


def test_current_upstream_signatures_are_mirrored():
    pairs = [
        (apriori, upstream_apriori),
        (fpgrowth, upstream_fpgrowth),
        (fpmax, upstream_fpmax),
        (association_rules, upstream_association_rules),
    ]
    for ours, theirs in pairs:
        assert list(inspect.signature(ours).parameters) == list(
            inspect.signature(theirs).parameters
        )


def sorted_rules(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_key"] = [
        (tuple(sorted(map(str, antecedent))), tuple(sorted(map(str, consequent))))
        for antecedent, consequent in zip(result["antecedents"], result["consequents"])
    ]
    return result.sort_values("_key").reset_index(drop=True)


@pytest.mark.parametrize(
    "metric,threshold",
    [
        ("support", 0.2),
        ("confidence", 0.55),
        ("lift", 0.9),
        ("leverage", -0.1),
        ("conviction", 0.5),
        ("zhangs_metric", -1.0),
        ("jaccard", 0.1),
        ("certainty", -1.0),
        ("kulczynski", 0.1),
    ],
)
def test_association_rules_all_threshold_metrics(metric, threshold):
    frequent = upstream_fpgrowth(basket(), min_support=0.125, use_colnames=True)
    ours = sorted_rules(
        association_rules(frequent, metric=metric, min_threshold=threshold)
    )
    theirs = sorted_rules(
        upstream_association_rules(frequent, metric=metric, min_threshold=threshold)
    )
    assert ours["_key"].tolist() == theirs["_key"].tolist()
    for column in [
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
    ]:
        assert np.allclose(ours[column], theirs[column], equal_nan=True, rtol=1e-14)


def test_association_rules_return_metrics_subset():
    frequent = upstream_apriori(basket(), min_support=0.125)
    columns = ["support", "confidence", "lift"]
    ours = association_rules(
        frequent, min_threshold=0.0, return_metrics=columns
    )
    theirs = upstream_association_rules(
        frequent, min_threshold=0.0, return_metrics=columns
    )
    assert list(ours.columns) == list(theirs.columns)
    ours, theirs = sorted_rules(ours), sorted_rules(theirs)
    assert ours["_key"].tolist() == theirs["_key"].tolist()
    assert np.allclose(ours[columns], theirs[columns])


def test_association_rules_support_only_with_incomplete_itemsets():
    frequent = pd.DataFrame(
        {"support": [0.4], "itemsets": [frozenset({"a", "b", "c"})]}
    )
    ours = sorted_rules(
        association_rules(
            frequent, support_only=True, min_threshold=0.3
        )
    )
    theirs = sorted_rules(
        upstream_association_rules(
            frequent, support_only=True, min_threshold=0.3
        )
    )
    assert ours["_key"].tolist() == theirs["_key"].tolist()
    assert np.allclose(ours["support"], theirs["support"])
    assert ours.drop(columns=["antecedents", "consequents", "support", "_key"]).isna().all().all()


def test_support_only_always_returns_support_column():
    frequent = pd.DataFrame(
        {"support": [0.4], "itemsets": [frozenset({"a", "b"})]}
    )
    ours = association_rules(
        frequent,
        support_only=True,
        min_threshold=0.0,
        return_metrics=["confidence"],
    )
    theirs = upstream_association_rules(
        frequent,
        support_only=True,
        min_threshold=0.0,
        return_metrics=["confidence"],
    )
    assert list(ours.columns) == list(theirs.columns)
    assert np.allclose(ours["support"], theirs["support"])
    assert ours["confidence"].isna().all()


def test_association_rules_missing_subsets_error():
    frequent = pd.DataFrame(
        {"support": [0.4], "itemsets": [frozenset({"a", "b"})]}
    )
    with pytest.raises(KeyError, match="support_only"):
        association_rules(frequent)


def test_null_value_mode_is_explicitly_out_of_scope():
    frame = pd.DataFrame({"a": [True, np.nan], "b": [False, True]})
    with pytest.raises(NotImplementedError, match="null_values"):
        fpgrowth(frame, null_values=True)
    with pytest.raises(NotImplementedError, match="null_values"):
        association_rules(
            pd.DataFrame({"support": [0.5], "itemsets": [frozenset({"a"})]}),
            null_values=True,
        )


def test_ffi_wrappers_reject_wrong_shapes_and_dtypes():
    with pytest.raises(TypeError, match="uint8"):
        _count_candidates(
            np.ones((2, 2), dtype=np.float64),
            np.array([[0]], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="non-empty-width"):
        _count_candidates(
            np.ones((2, 2), dtype=np.uint8),
            np.empty((1, 0), dtype=np.int64),
        )
    with pytest.raises(TypeError, match="parent"):
        _intersect_items(
            np.ones((2, 2), dtype=np.uint64),
            np.array([0], dtype=np.int64),
            np.ones(3, dtype=np.uint64),
            1,
        )
    with pytest.raises(ValueError, match="three columns"):
        _rule_metric_matrix(np.ones((2, 2), dtype=np.float64))


def test_mojo_exports_report_invalid_buffers_and_indices():
    library = lib()
    assert library.mmlx_count_candidates(0, 0, 0, 1, 1, 1, 1, 1) != 0
    assert library.mmlx_rule_metrics(0, 0, 1, 1) != 0

    vertical = np.ones((1, 1), dtype=np.uint64)
    item_ids = np.array([1], dtype=np.int64)
    children = np.empty((1, 1), dtype=np.uint64)
    counts = np.empty(1, dtype=np.int64)
    status = library.mmlx_intersect_count(
        0,
        vertical.ctypes.data,
        item_ids.ctypes.data,
        children.ctypes.data,
        counts.ctypes.data,
        1,
        1,
        1,
        1,
    )
    assert status != 0
