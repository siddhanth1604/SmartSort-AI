"""All algorithms are written in pure Python so the timing comparison is fair.
(Using the built-in sorted() would always win, because it is implemented in C.)"""


def insertion_sort(arr):

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def quick_sort(arr):
    """Textbook in-place quicksort: Lomuto partition, last element as pivot.
    Very fast on shuffled data, but degrades to O(n^2) on sorted, reversed or
    heavily repeated values (and may hit Python's recursion limit). Those weak
    spots are exactly where a selector should pick something else."""

    def sort(lo, hi):
        if lo < hi:
            pivot = arr[hi]
            i = lo
            for j in range(lo, hi):
                if arr[j] < pivot:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
            arr[i], arr[hi] = arr[hi], arr[i]
            sort(lo, i - 1)
            sort(i + 1, hi)

    sort(0, len(arr) - 1)
    return arr


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def heap_sort(arr):

    n = len(arr)

    def sift_down(root, end):
        while True:
            child = 2 * root + 1
            if child >= end:
                return
            if child + 1 < end and arr[child] < arr[child + 1]:
                child += 1
            if arr[root] < arr[child]:
                arr[root], arr[child] = arr[child], arr[root]
                root = child
            else:
                return

    for start in range(n // 2 - 1, -1, -1):
        sift_down(start, n)
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        sift_down(0, end)
    return arr


def counting_sort(arr):
    if len(arr) == 0:
        return arr
    min_val, max_val = min(arr), max(arr)
    if max_val - min_val > 5_000_000:
        raise ValueError("value range too large for counting sort")
    count = [0] * (max_val - min_val + 1)
    for a in arr:
        count[a - min_val] += 1
    i = 0
    for value, c in enumerate(count):
        for _ in range(c):
            arr[i] = value + min_val
            i += 1
    return arr


def _insertion_range(arr, lo, hi, start):
    for k in range(start, hi):
        key = arr[k]
        j = k - 1
        while j >= lo and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def _merge_runs(arr, lo, mid, hi):
    left, right = arr[lo:mid], arr[mid:hi]
    i = j = 0
    k = lo
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


def tim_sort(arr):
    """Simplified TimSort: find natural runs (reversing descending ones),
    extend short runs with insertion sort, then merge runs pairwise."""
    n = len(arr)
    if n < 2:
        return arr
    r, m = 0, n
    while m >= 64:
        r |= m & 1
        m >>= 1
    minrun = m + r

    runs, i = [], 0
    while i < n:
        j = i + 1
        if j < n:
            if arr[j] < arr[i]:
                while j < n and arr[j] < arr[j - 1]:
                    j += 1
                arr[i:j] = arr[i:j][::-1]
            else:
                while j < n and arr[j] >= arr[j - 1]:
                    j += 1
        end = min(n, max(j, i + minrun))
        if end > j:
            _insertion_range(arr, i, end, j)
        runs.append((i, end))
        i = end

    while len(runs) > 1:
        merged = []
        for k in range(0, len(runs) - 1, 2):
            (a, b), (_, c) = runs[k], runs[k + 1]
            _merge_runs(arr, a, b, c)
            merged.append((a, c))
        if len(runs) % 2:
            merged.append(runs[-1])
        runs = merged
    return arr

ALGORITHMS = {
    "InsertionSort": insertion_sort,
    "QuickSort": quick_sort,
    "HeapSort": heap_sort,
    "CountingSort": counting_sort,
    "TimSort": tim_sort,
}
