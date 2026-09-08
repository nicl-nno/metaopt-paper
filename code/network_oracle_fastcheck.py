"""Exact faster oracle check, separate from the frozen pilot implementation.

The degree attack order does not depend on component sizes. Compute that order
first, then reconstruct every component size backwards with union-find.
"""
import heapq
import json
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).parent/"GAMLET"
sys.path.insert(0, str(REPO))
from experiments.network_robustness.problem import Task, NetworkDesign, bit_rows, robustness


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


def run(path):
    cfg = {"attack_scenarios": 4, "max_changed_fraction": .2}
    measurements = []
    for n in [12, 50, 100, 200]:
        for family in ("er", "ba"):
            problem = NetworkDesign(Task("calibration", "calibration", family, n, 533+n, 893+n), cfg)
            rng = np.random.default_rng(111)
            for sample in range(3):
                adj = problem.sample(rng)
                start = time.perf_counter()
                old_r, old_curves = robustness(adj, problem.priorities, True)
                old_s = time.perf_counter()-start
                start = time.perf_counter()
                new_r, new_curves = fast_robustness(adj, problem.priorities)
                new_s = time.perf_counter()-start
                np.testing.assert_array_equal(old_curves, new_curves)
                assert old_r == new_r
                measurements.append({"n": n, "family": family, "sample": sample,
                                     "pilot_seconds": old_s, "union_find_seconds": new_s})
    path.write_text(json.dumps({"all_exactly_equal": True, "checked_graphs": len(measurements),
                     "attack_trajectories": len(measurements)*4, "measurements": measurements,
                     "scope": "New calibration graphs only, no selection or alteration of pilot results"}, indent=2))
    for n in [12, 50, 100, 200]:
        group = [m for m in measurements if m["n"] == n]
        print(n, np.median([m["pilot_seconds"] for m in group]), np.median([m["union_find_seconds"] for m in group]))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
