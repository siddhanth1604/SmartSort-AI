import numpy as np

def insertion_sort(arr):
    # Best for small or nearly sorted arrays
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def quick_sort(arr):
    # General purpose comparison sort
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr):
    # O(n log n) with excellent memory footprint
    import heapq
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

def counting_sort(arr):
    if len(arr) == 0: return arr
    
    min_val = min(arr)
    max_val = max(arr)
    range_of_elements = max_val - min_val + 1
    
    count = [0] * range_of_elements
    for a in arr:
        count[a - min_val] += 1
        
    i = 0
    for a in range(range_of_elements):
        for _ in range(count[a]):
            arr[i] = a + min_val
            i += 1
    return arr

def tim_sort(arr):
    return sorted(arr)

ALGORITHMS = {
    "InsertionSort": insertion_sort,
    "QuickSort": quick_sort,
    "HeapSort": heap_sort,
    "CountingSort": counting_sort,
    "TimSort": tim_sort
}