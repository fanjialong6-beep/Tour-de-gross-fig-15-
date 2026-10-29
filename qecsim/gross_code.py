from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Reduce a binary matrix to reduced row echelon form and return its pivot columns.
def rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    a = np.asarray(matrix, dtype=np.uint8).copy() & 1
    pivots: list[int] = []
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[len(pivots):, col])
        if not len(candidates):
            continue
        row = len(pivots)
        other = row + int(candidates[0])
        a[[row, other]] = a[[other, row]]
        eliminate = np.flatnonzero(a[:, col])
        eliminate = eliminate[eliminate != row]
        a[eliminate] ^= a[row]
        pivots.append(col)
        if len(pivots) == a.shape[0]:
            break
    return a, pivots


# Compute the rank of a binary matrix over GF(2).
def rank(matrix: np.ndarray) -> int:
    return len(rref(matrix)[1])


# Select independent rows of a binary matrix.
def independent_rows(matrix: np.ndarray) -> np.ndarray:
    basis: list[np.ndarray] = []
    current = np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    current_rank = 0
    for row in np.asarray(matrix, dtype=np.uint8):
        trial = np.vstack([current, row])
        trial_rank = rank(trial)
        if trial_rank > current_rank:
            basis.append(row.copy())
            current = trial
            current_rank = trial_rank
    return np.asarray(basis, dtype=np.uint8)


# Construct a basis for the binary nullspace.
def nullspace(matrix: np.ndarray) -> np.ndarray:
    a, pivots = rref(matrix)
    free = [j for j in range(a.shape[1]) if j not in pivots]
    basis = np.zeros((len(free), a.shape[1]), dtype=np.uint8)
    for i, col in enumerate(free):
        basis[i, col] = 1
        basis[i, pivots] = a[: len(pivots), col]
    return basis


# Complete stabilizers to a basis of the commuting operator space.
def quotient_basis(kernel_check: np.ndarray, stabilizers: np.ndarray) -> np.ndarray:
    pivots: dict[int, int] = {}

    # Insert an independent vector into the binary elimination basis.
    def insert(row: np.ndarray) -> bool:
        value = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                return True
            value ^= pivots[pivot]
        return False

    for row in stabilizers:
        insert(row)
    result = [row for row in nullspace(kernel_check) if insert(row)]
    return np.asarray(result, dtype=np.uint8).reshape(-1, kernel_check.shape[1])


# Invert a square binary matrix over GF(2).
def inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    n = matrix.shape[0]
    aug = np.hstack([matrix.copy(), np.eye(n, dtype=np.uint8)])
    for col in range(n):
        rows = np.flatnonzero(aug[col:, col])
        if not len(rows):
            raise ValueError("matrix is singular over GF(2)")
        pivot = col + int(rows[0])
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in np.flatnonzero(aug[:, col]):
            if row != col:
                aug[row] ^= aug[col]
    return aug[:, n:]


@dataclass(frozen=True)
class GrossCode:
    ell: int
    m: int
    hx: np.ndarray
    hz: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray

    # Return the number of physical qubits.
    @property
    def n(self) -> int:
        return 2 * self.ell * self.m

    # Map periodic lattice coordinates to a physical qubit index.
    def index(self, block: int, x: int, y: int) -> int:
        return block * self.ell * self.m + (x % self.ell) * self.m + y % self.m

    # Convert a physical qubit index to its lattice label.
    def label(self, index: int) -> tuple[str, int, int]:
        block, offset = divmod(index, self.ell * self.m)
        x, y = divmod(offset, self.m)
        return ("L" if block == 0 else "R", x, y)


# Build a binary periodic shift matrix from polynomial terms.
def _poly_matrix(ell: int, m: int, terms: list[tuple[int, int]]) -> np.ndarray:
    size = ell * m
    result = np.zeros((size, size), dtype=np.uint8)
    for x in range(ell):
        for y in range(m):
            for dx, dy in terms:
                result[x * m + y, ((x + dx) % ell) * m + (y + dy) % m] ^= 1
    return result


# Build a binary operator from left and right lattice supports.
def _vector(code: GrossCode, left: list[tuple[int, int]], right: list[tuple[int, int]]) -> np.ndarray:
    result = np.zeros(code.n, dtype=np.uint8)
    for block, terms in enumerate((left, right)):
        for x, y in terms:
            result[code.index(block, x, y)] ^= 1
    return result


# Construct the [[144,12,12]] gross-code checks and a logical basis.
def make_gross_code() -> GrossCode:
    ell, m = 12, 6
    a = _poly_matrix(ell, m, [(0, 0), (0, 1), (3, -1)])
    b = _poly_matrix(ell, m, [(0, 0), (1, 0), (-1, -3)])
    hx = np.hstack([a, b])
    hz = np.hstack([b.T, a.T])

    raw_x = quotient_basis(hz, hx)
    raw_z = quotient_basis(hx, hz)
    temporary = GrossCode(ell, m, hx, hz, raw_x, raw_z)

    p = [(4, 0), (5, 0), (6, 1), (4, 2), (5, 4), (6, 5)]
    q = [(3, 0), (4, 0), (3, 1), (3, 2), (4, 2), (3, 5)]
    r = [(0, 0), (8, 0), (1, 1), (9, 1), (3, 4), (11, 4)]
    s = [(1, 0), (9, 0), (4, 4), (8, 4), (0, 5), (8, 5)]
    x1 = _vector(temporary, p, q)
    x7 = _vector(temporary, r, s)

    logical_x_rows = [x1, x7]
    span = np.vstack([independent_rows(hx), x1, x7])
    span_rank = rank(span)
    for candidate in raw_x:
        trial = np.vstack([span, candidate])
        new_rank = rank(trial)
        if new_rank > span_rank:
            logical_x_rows.append(candidate.copy())
            span = trial
            span_rank = new_rank
        if len(logical_x_rows) == 12:
            break
    logical_x = np.asarray(logical_x_rows, dtype=np.uint8)

    pairing = (logical_x @ raw_z.T) & 1
    logical_z = (inverse(pairing).T @ raw_z) & 1
    assert np.array_equal((logical_x @ logical_z.T) & 1, np.eye(12, dtype=np.uint8))
    assert rank(hx) == rank(hz) == 66
    return GrossCode(ell, m, hx, hz, logical_x, logical_z)


# Choose logical generators for the requested X measurement.
def measurement_basis(code: GrossCode, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "X":
        transform = np.eye(12, dtype=np.uint8)
    elif mode == "XX":
        transform = np.eye(12, dtype=np.uint8)
        transform[0] ^= transform[1]
    else:
        raise ValueError(mode)
    x_basis = (transform @ code.logical_x) & 1
    z_basis = (inverse(transform).T @ code.logical_z) & 1
    assert np.array_equal((x_basis @ z_basis.T) & 1, np.eye(12, dtype=np.uint8))
    return x_basis, z_basis
