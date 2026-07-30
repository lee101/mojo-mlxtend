"""Frequent-pattern counting and association-rule metric kernels."""

from std.algorithm.functional import parallelize
from std.bit import pop_count
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime UW = simd_width_of[DType.uint64]()


def count_candidate(
    data: BPtr,
    candidates: IPtr,
    counts: IPtr,
    candidate: Int,
    rows: Int,
    cols: Int,
    width: Int,
):
    var count = 0
    var candidate_offset = candidate * width
    for row in range(rows):
        var present = True
        var column = 0
        while column < width:
            var item = Int(candidates[candidate_offset + column])
            if item < 0 or item >= cols or data[row * cols + item] == 0:
                present = False
                break
            column += 1
        if present:
            count += 1
    counts[candidate] = Int64(count)


@export("mmlx_count_candidates")
def mmlx_count_candidates(
    data_addr: Int,
    candidates_addr: Int,
    counts_addr: Int,
    rows: Int,
    cols: Int,
    candidate_count: Int,
    width: Int,
    threads: Int,
) abi("C") -> Int64:
    if (
        data_addr == 0
        or candidates_addr == 0
        or counts_addr == 0
        or rows < 0
        or cols < 0
        or candidate_count <= 0
        or width <= 0
        or threads <= 0
    ):
        return -1
    var data = BPtr(unsafe_from_address=data_addr)
    var candidates = IPtr(unsafe_from_address=candidates_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)

    @parameter
    def work(candidate: Int):
        count_candidate(data, candidates, counts, candidate, rows, cols, width)

    if threads > 1 and candidate_count * rows * width >= 1_000_000:
        parallelize[work](candidate_count, min(threads, candidate_count))
    else:
        for candidate in range(candidate_count):
            work(candidate)
    return 0


def copy_and_count(source: UPtr, destination: UPtr, words: Int) -> Int:
    var count = 0
    var word = 0
    while word + UW <= words:
        var values = source.load[width=UW](word)
        destination.store(word, values)
        count += Int(pop_count(values).reduce_add())
        word += UW
    while word < words:
        var value = source[word]
        destination[word] = value
        count += Int(pop_count(value))
        word += 1
    return count


def intersect_and_count(
    parent: UPtr, source: UPtr, destination: UPtr, words: Int
) -> Int:
    var count = 0
    var word = 0
    while word + UW <= words:
        var values = (
            parent.load[width=UW](word) & source.load[width=UW](word)
        )
        destination.store(word, values)
        count += Int(pop_count(values).reduce_add())
        word += UW
    while word < words:
        var value = parent[word] & source[word]
        destination[word] = value
        count += Int(pop_count(value))
        word += 1
    return count


@export("mmlx_intersect_count")
def mmlx_intersect_count(
    parent_addr: Int,
    vertical_addr: Int,
    item_ids_addr: Int,
    child_addr: Int,
    counts_addr: Int,
    item_count: Int,
    vertical_items: Int,
    words: Int,
    threads: Int,
) abi("C") -> Int64:
    if (
        vertical_addr == 0
        or item_ids_addr == 0
        or child_addr == 0
        or counts_addr == 0
        or item_count <= 0
        or vertical_items <= 0
        or words <= 0
        or threads <= 0
    ):
        return -1
    var vertical = UPtr(unsafe_from_address=vertical_addr)
    var item_ids = IPtr(unsafe_from_address=item_ids_addr)
    var children = UPtr(unsafe_from_address=child_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    for index in range(item_count):
        var item = Int(item_ids[index])
        if item < 0 or item >= vertical_items:
            return -2
    if parent_addr == 0:

        @parameter
        def root_work(index: Int):
            var item = Int(item_ids[index])
            var count = copy_and_count(
                vertical + item * words, children + index * words, words
            )
            counts[index] = Int64(count)

        if threads > 1 and item_count * words >= 16_384:
            parallelize[root_work](item_count, min(threads, item_count))
        else:
            for index in range(item_count):
                root_work(index)
    else:
        var parent = UPtr(unsafe_from_address=parent_addr)

        @parameter
        def child_work(index: Int):
            var item = Int(item_ids[index])
            var count = intersect_and_count(
                parent,
                vertical + item * words,
                children + index * words,
                words,
            )
            counts[index] = Int64(count)

        if threads > 1 and item_count * words >= 16_384:
            parallelize[child_work](item_count, min(threads, item_count))
        else:
            for index in range(item_count):
                child_work(index)
    return 0


def infinity() -> Float64:
    var zero = 0.0
    return 1.0 / zero


def compute_metric_row(source: FPtr, destination: FPtr, row: Int):
    var combined = source[row * 3]
    var antecedent = source[row * 3 + 1]
    var consequent = source[row * 3 + 2]
    var confidence = combined / antecedent
    var lift = confidence / consequent
    var leverage = combined - antecedent * consequent
    var zhang_denominator = max(
        combined * (1.0 - antecedent),
        antecedent * (consequent - combined),
    )
    var jaccard_denominator = antecedent + consequent - combined
    var certainty_denominator = 1.0 - consequent
    var offset = row * 12
    destination[offset] = antecedent
    destination[offset + 1] = consequent
    destination[offset + 2] = combined
    destination[offset + 3] = confidence
    destination[offset + 4] = lift
    destination[offset + 5] = 1.0
    destination[offset + 6] = leverage
    destination[offset + 7] = (
        infinity()
        if confidence >= 1.0
        else (1.0 - consequent) / (1.0 - confidence)
    )
    destination[offset + 8] = (
        0.0 if zhang_denominator == 0.0 else leverage / zhang_denominator
    )
    destination[offset + 9] = (
        0.0 if jaccard_denominator == 0.0 else combined / jaccard_denominator
    )
    destination[offset + 10] = (
        0.0
        if certainty_denominator == 0.0
        else (confidence - consequent) / certainty_denominator
    )
    destination[offset + 11] = (
        combined / antecedent + combined / consequent
    ) * 0.5


@export("mmlx_rule_metrics")
def mmlx_rule_metrics(
    source_addr: Int,
    destination_addr: Int,
    rows: Int,
    threads: Int,
) abi("C") -> Int64:
    if source_addr == 0 or destination_addr == 0 or rows <= 0 or threads <= 0:
        return -1
    var source = FPtr(unsafe_from_address=source_addr)
    var destination = FPtr(unsafe_from_address=destination_addr)

    @parameter
    def work(row: Int):
        compute_metric_row(source, destination, row)

    if threads > 1 and rows >= 16_384:
        parallelize[work](rows, min(threads, rows))
    else:
        for row in range(rows):
            work(row)
    return 0
