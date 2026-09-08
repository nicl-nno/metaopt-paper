"""Pin-jointed, small-displacement 2-D truss compliance at fixed material volume.

SI units. All left-boundary translations are fixed. Mandatory triangular cells
make every encoded design stable; optional opposite diagonals change topology.
Crossing bars share a joint only at an explicitly defined endpoint. This is a
numerical benchmark, not a fabrication or structural certification procedure.
"""

from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd

import numpy as np


def solve_axial_system(b_free, axial_stiffness, force_free):
    stiffness = b_free.T @ (axial_stiffness[:, None] * b_free)
    displacement = np.linalg.solve(stiffness, force_free)
    compliance = float(force_free @ displacement)
    residual = np.linalg.norm(stiffness @ displacement-force_free) / np.linalg.norm(force_free)
    if not np.isfinite(compliance) or compliance <= 0 or residual > 1e-7:
        raise ArithmeticError("Invalid mechanical evaluation")
    return displacement, compliance, float(residual)


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    span: float
    height: float
    force: float
    angle: float
    upper_fraction: float

    def to_dict(self):
        return asdict(self)


class Truss:
    def __init__(self, task: Task, nx=4, ny=3, volume=0.03, young=210e9):
        self.task, self.nx, self.ny = task, nx, ny
        self.volume, self.young = float(volume), float(young)
        self.xy = np.array([(x * task.span / (nx - 1), y * task.height / (ny - 1))
                            for y in range(ny) for x in range(nx)], dtype=float)
        ix = lambda x, y: y * nx + x
        horizontal = [(ix(x, y), ix(x + 1, y)) for y in range(ny) for x in range(nx - 1)]
        vertical = [(ix(x, y), ix(x, y + 1)) for y in range(ny - 1) for x in range(nx)]
        diagonal = [(ix(x, y), ix(x + 1, y + 1)) for y in range(ny - 1) for x in range(nx - 1)]
        optional = [(ix(x + 1, y), ix(x, y + 1)) for y in range(ny - 1) for x in range(nx - 1)]
        self.edges = np.array(horizontal + vertical + diagonal + optional)
        self.mandatory = len(horizontal + vertical + diagonal)
        self.n_members = len(self.edges)
        delta = self.xy[self.edges[:, 1]] - self.xy[self.edges[:, 0]]
        self.length = np.linalg.norm(delta, axis=1)
        self.direction = delta / self.length[:, None]
        self.b = np.zeros((self.n_members, self.xy.size))
        for i, (a, b) in enumerate(self.edges):
            self.b[i, 2*a:2*a+2] = -self.direction[i]
            self.b[i, 2*b:2*b+2] = self.direction[i]
        self.fixed = np.repeat(np.isclose(self.xy[:, 0], 0), 2)
        self.free = ~self.fixed
        self.b_free = self.b[:, self.free]
        self.force_vector = np.zeros(self.xy.size)
        # angle=0 means downward loading; magnitude changes do not alter units.
        vector = task.force * np.array([np.sin(task.angle), -np.cos(task.angle)])
        for node, fraction in [(ix(nx-1, ny-1), task.upper_fraction),
                               (ix(nx-1, 0), 1-task.upper_fraction)]:
            self.force_vector[2*node:2*node+2] += fraction * vector
        self.f_free = self.force_vector[self.free]
        incidence = np.zeros((self.n_members, len(self.xy)))
        incidence[np.arange(self.n_members), self.edges[:, 0]] = 1
        incidence[np.arange(self.n_members), self.edges[:, 1]] = 1
        self.line_adjacency = ((incidence @ incidence.T) > 0).astype(np.float32)
        np.fill_diagonal(self.line_adjacency, 0)

    @staticmethod
    def key(genes):
        # Proportional positive areas describe the same fixed-volume structure.
        values = tuple(int(v) for v in genes)
        divisor = reduce(gcd, values)
        if divisor == 0:
            raise ValueError("Empty design")
        return tuple(v // divisor for v in values)

    def sample(self, rng):
        genes = rng.integers(1, 5, self.n_members)
        genes[self.mandatory:] = rng.integers(0, 5, self.n_members-self.mandatory)
        return genes.astype(np.int8)

    def areas(self, genes):
        genes = np.asarray(genes, dtype=float)
        if genes.shape != (self.n_members,) or not np.isfinite(genes).all():
            raise ValueError("Invalid design shape or nonfinite genes")
        if np.any(genes[:self.mandatory] <= 0) or np.any(genes < 0):
            raise ValueError("Mandatory members must have positive area")
        return self.volume * genes / (self.length @ genes)

    def solve(self, genes):
        areas = self.areas(genes)
        displacement_free, compliance, residual = solve_axial_system(
            self.b_free, self.young * areas / self.length, self.f_free)
        displacement = np.zeros(self.xy.size)
        displacement[self.free] = displacement_free
        return {"compliance": compliance, "volume": float(areas @ self.length),
                "active_members": int(np.count_nonzero(areas)),
                "residual": float(residual), "displacement": displacement,
                "areas": areas}

    def features(self, genes):
        """Member-as-node line graph; no solver outputs enter model features."""
        areas = self.areas(genes)
        mask = (areas > 0).astype(np.float32)
        a, b = self.edges.T
        coordinates = self.xy / np.array([self.task.span, self.task.height])
        forces = self.force_vector.reshape(-1, 2) / 1e5
        supports = self.fixed.reshape(-1, 2).astype(float)
        nodes = np.column_stack([
            coordinates[a], coordinates[b], self.direction,
            self.length / 6, areas / 0.003, areas * self.length / self.volume,
            forces[a], forces[b], supports[a], supports[b],
        ]).astype(np.float32)
        adjacency = self.line_adjacency * mask[:, None] * mask[None, :]
        adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1)
        context = np.array([self.task.span/6, self.task.height/3, self.task.force/1e5,
                            np.sin(self.task.angle), np.cos(self.task.angle),
                            self.task.upper_fraction], dtype=np.float32)
        return nodes, adjacency, mask, context


def make_tasks(split, count, seed):
    rng = np.random.default_rng(seed)
    return [Task(f"{split}_{i:03d}", split, float(rng.uniform(3, 6)),
                 float(rng.uniform(1, 2.5)), float(rng.uniform(0.5e5, 1.5e5)),
                 float(rng.uniform(-0.35, 0.35)), float(rng.uniform(0.15, 0.85)))
            for i in range(count)]
