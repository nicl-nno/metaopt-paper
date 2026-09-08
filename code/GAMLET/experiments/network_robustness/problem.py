"""Connected simple graph design under adaptive highest-degree node removal."""
from dataclasses import asdict, dataclass
import math

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    family: str
    n: int
    graph_seed: int
    attack_seed: int

    def to_dict(self):
        return asdict(self)


def make_tasks(config):
    rng = np.random.default_rng(config["seed"])
    tasks = []
    for split, count in config["tasks_per_cell"].items():
        for family in ("er", "ba"):
            for n in config.get("split_sizes", {}).get(split, config["sizes"]):
                for i in range(count):
                    tasks.append(Task(f"{split}_{family}_{n}_{i:02d}", split, family, n,
                                      int(rng.integers(2**30)), int(rng.integers(2**30))))
    for family in ("er", "ba"):
        for n in config.get("extrapolation_sizes", []):
            for i in range(config.get("extrapolation_tasks_per_cell", 0)):
                tasks.append(Task(f"test_{family}_{n}_{i:02d}", "test", family, n,
                                  int(rng.integers(2**30)), int(rng.integers(2**30))))
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs; extrapolation sizes must differ from test sizes")
    return tasks


def initial_graph(task):
    rng = np.random.default_rng(task.graph_seed)
    for _ in range(10000):
        seed = int(rng.integers(2**30))
        if task.family == "ba":
            graph = nx.barabasi_albert_graph(task.n, 3, seed=seed)
        else:
            # Matched edge count, ER G(n,m) conditioned on connectivity.
            graph = nx.gnm_random_graph(task.n, 3*(task.n-3), seed=seed)
        if nx.is_connected(graph):
            # Remove incidental BA age ordering; node IDs never enter features.
            p = rng.permutation(task.n)
            adj = nx.to_numpy_array(graph, dtype=bool)
            return adj[np.ix_(p, p)]
    raise RuntimeError("Could not generate connected graph")


def bit_rows(adj):
    return [sum(1 << int(j) for j in np.flatnonzero(row)) for row in adj]


def largest_component(rows, active):
    unseen, best = active, 0
    while unseen:
        frontier = unseen & -unseen
        unseen ^= frontier
        size = 0
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            size += 1
            neighbors = rows[bit.bit_length()-1] & unseen
            unseen ^= neighbors
            frontier |= neighbors
        best = max(best, size)
        if unseen.bit_count() <= best:
            break
    return best


def attack_priorities(n, seed, count):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(count)])


def robustness(adj, priorities, return_curves=False):
    """R = sum_{q=1}^n |LCC after q deletions| / n^2, averaged over ties.

    Degrees are recomputed by exact decrements after EVERY deletion. At ties,
    the surviving node with the smallest scenario priority is removed. This
    fixed scenario bank is common across candidates and algorithms in a task.
    """
    n = len(adj)
    rows = bit_rows(adj)
    original_degree = adj.sum(1).astype(np.int64)
    curves = []
    for priority in priorities:
        degree = original_degree.copy()
        active = (1 << n)-1
        curve = []
        for _ in range(n):
            ties = np.flatnonzero(degree == degree.max())
            node = int(ties[np.argmin(priority[ties])])
            active &= ~(1 << node)
            degree[node] = -1
            neighbors = rows[node] & active
            while neighbors:
                bit = neighbors & -neighbors
                degree[bit.bit_length()-1] -= 1
                neighbors ^= bit
            curve.append(largest_component(rows, active))
        curves.append(curve)
    curves = np.array(curves, dtype=np.int64)
    value = float(curves.sum(1).mean()/n**2)
    return (value, curves) if return_curves else value


def graph_key(adj):
    return len(adj).to_bytes(4, "little")+np.packbits(adj[np.triu_indices(len(adj), 1)]).tobytes()


class NetworkDesign:
    def __init__(self, task, config):
        self.task = task
        self.oracle = config.get("oracle", "bfs")
        if self.oracle not in ("bfs", "union_find"):
            raise ValueError("oracle must be bfs or union_find")
        self.base = initial_graph(task)
        self.degree = self.base.sum(1)
        self.m = int(self.degree.sum()//2)
        self.max_new_edges = max(2, int(math.ceil(config["max_changed_fraction"]*self.m)))
        self.priorities = attack_priorities(task.n, task.attack_seed, config["attack_scenarios"])

    def valid(self, adj):
        if adj.shape != self.base.shape or not np.array_equal(adj, adj.T) or adj.diagonal().any():
            return False
        if not np.array_equal(adj.sum(1), self.degree):
            return False
        if np.count_nonzero(adj & ~self.base)//2 > self.max_new_edges:
            return False
        return largest_component(bit_rows(adj), (1 << len(adj))-1) == len(adj)

    def swap(self, adj, rng):
        edges = np.column_stack(np.where(np.triu(adj, 1)))
        for _ in range(500):
            selected = edges[rng.choice(len(edges), size=2, replace=False)]
            a, b = map(int, selected[0])
            c, d = map(int, selected[1])
            if len({a, b, c, d}) < 4:
                continue
            if rng.integers(2):
                c, d = d, c
            if adj[a, c] or adj[b, d]:
                continue
            result = adj.copy()
            result[a, b] = result[b, a] = result[c, d] = result[d, c] = False
            result[a, c] = result[c, a] = result[b, d] = result[d, b] = True
            if self.valid(result):
                return result
        raise RuntimeError("No feasible double-edge swap found")

    def sample(self, rng):
        adj = self.base.copy()
        for _ in range(int(rng.integers(1, self.max_new_edges+1))):
            adj = self.swap(adj, rng)
        return adj

    def evaluate(self, adj):
        if not self.valid(adj):
            raise ValueError("Invalid network design")
        if self.oracle == "union_find":
            from .fast_oracle import fast_robustness
            return fast_robustness(adj, self.priorities)[0]
        return robustness(adj, self.priorities)

    def features(self, adj):
        n = len(adj)
        degree = adj.sum(1).astype(float)
        mean = degree.mean()
        neighbor_mean = adj @ degree / np.maximum(degree, 1)
        neighbor_var = np.maximum(adj @ (degree**2)/np.maximum(degree, 1)-neighbor_mean**2, 0)
        rows = bit_rows(adj)
        clustering = np.zeros(n)
        for i, row in enumerate(rows):
            if degree[i] < 2:
                continue
            neighbors, doubled_triangles = row, 0
            while neighbors:
                bit = neighbors & -neighbors
                doubled_triangles += (rows[bit.bit_length()-1] & row).bit_count()
                neighbors ^= bit
            clustering[i] = doubled_triangles/(degree[i]*(degree[i]-1))
        nodes = np.column_stack((degree/(n-1), degree/mean, neighbor_mean/mean,
                                 np.sqrt(neighbor_var)/mean, clustering, np.ones(n))).astype(np.float32)
        context = np.array([n/200, mean/10, degree.max()/n, degree.std()/mean,
                            self.max_new_edges/self.m, 1], dtype=np.float32)
        return nodes, adj.astype(np.float32), np.ones(n, dtype=np.float32), context


def propose(problem, parents, count, rng, excluded, restart=False, greedy=False):
    pool, seen = [], set(excluded)
    for _ in range(20000):
        if restart:
            candidate = problem.sample(rng)
        else:
            candidate = parents[int(rng.integers(len(parents)))].copy()
            for _ in range(1 if greedy else int(rng.integers(1, 4))):
                candidate = problem.swap(candidate, rng)
        key = graph_key(candidate)
        if key not in seen:
            seen.add(key)
            pool.append(candidate)
            if len(pool) == count:
                return pool
    raise RuntimeError("Candidate pool exhausted")
