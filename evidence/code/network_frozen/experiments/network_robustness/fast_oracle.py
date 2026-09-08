# Exact adaptive-degree robustness via lazy heap and reverse union-find.
import heapq
import numpy as np
from .problem import bit_rows

def fast_robustness(adj, priorities):
    n = len(adj)
    rows = bit_rows(adj)
    curves = []
    for priority in priorities:
        degree = adj.sum(1).tolist()
        active = [True]*n
        heap = [(-degree[i], int(priority[i]), i) for i in range(n)]
        heapq.heapify(heap)
        order = []
        while heap:
            neg_degree, _, node = heapq.heappop(heap)
            if not active[node] or -neg_degree != degree[node]:
                continue
            order.append(node)
            active[node] = False
            remaining = rows[node]
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length()-1
                remaining ^= bit
                if active[j]:
                    degree[j] -= 1
                    heapq.heappush(heap, (-degree[j], int(priority[j]), j))
        assert len(order) == n
        parent = list(range(n))
        size = [1]*n
        present = [False]*n
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        maximum = 0
        curve = [0]*n
        for i in range(n-1, -1, -1):
            curve[i] = maximum
            node = order[i]
            present[node] = True
            maximum = max(maximum, 1)
            remaining = rows[node]
            while remaining:
                bit = remaining & -remaining
                j = bit.bit_length()-1
                remaining ^= bit
                if present[j]:
                    a, b = root(node), root(j)
                    if a != b:
                        if size[a] < size[b]:
                            a, b = b, a
                        parent[b] = a
                        size[a] += size[b]
                        maximum = max(maximum, size[a])
        curves.append(curve)
    curves = np.array(curves)
    return float(curves.sum(1).mean()/n**2), curves

