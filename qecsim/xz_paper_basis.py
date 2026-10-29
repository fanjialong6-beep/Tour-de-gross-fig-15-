from __future__ import annotations

import numpy as np
from .gross_code import GrossCode, make_gross_code, rank

PAPER_LOGICAL_ORDER = (1, 7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
ALPHA = ((0, 0), (3, 5), (11, 5), (10, 1), (5, 4), (4, 2))
BETA = ((0, 0), (2, 4), (1, 2), (2, 5), (1, 1), (3, 1))


# Translate a logical support on the periodic gross-code lattice.
def shift(code: GrossCode, row: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros(code.n, dtype=np.uint8)
    for q in np.flatnonzero(row):
        block, remainder = divmod(int(q), code.ell * code.m)
        x, y = divmod(remainder, code.m)
        out[code.index(block, x + dx, y + dy)] ^= 1
    return out


# Map one qubit through the paper duality transformation.
def paper_dual_index(code: GrossCode, q: int) -> int:
    block, remainder = divmod(int(q), code.ell * code.m)
    x, y = divmod(remainder, code.m)
    return code.index(1 - block, 1 - x, 1 - y)


# Apply the paper duality transformation to an operator support.
def paper_dual(code: GrossCode, row: np.ndarray) -> np.ndarray:
    out = np.zeros(code.n, dtype=np.uint8)
    for q in np.flatnonzero(row):
        out[paper_dual_index(code, int(q))] ^= 1
    return out


# Construct the paper logical basis in the order 1,7,2,3,4,5,6,8,9,10,11,12.
def make_paper_gross_code() -> GrossCode:
    base = make_gross_code()
    x1, x7 = base.logical_x[:2]
    z1, z7 = paper_dual(base, x7), paper_dual(base, x1)
    xs = [shift(base, x1, *a) for a in ALPHA]
    xs += [shift(base, x7, -b[0], -b[1]) for b in BETA]
    zs = [shift(base, z1, *b) for b in BETA]
    zs += [shift(base, z7, -a[0], -a[1]) for a in ALPHA]
    order = np.asarray(PAPER_LOGICAL_ORDER) - 1
    code = GrossCode(base.ell, base.m, base.hx, base.hz,
                     np.asarray(xs, dtype=np.uint8)[order],
                     np.asarray(zs, dtype=np.uint8)[order])
    assert np.array_equal(code.logical_x @ code.logical_z.T % 2, np.eye(12, dtype=np.uint8))
    assert not np.any(code.hz @ code.logical_x.T % 2)
    assert not np.any(code.hx @ code.logical_z.T % 2)
    assert rank(np.vstack([code.hx, code.logical_x])) == 78
    assert rank(np.vstack([code.hz, code.logical_z])) == 78
    assert np.all(code.logical_x.sum(axis=1) == 12)
    assert np.all(code.logical_z.sum(axis=1) == 12)
    return code


# Summarize the explicit logical supports and their symplectic pairing.
def paper_basis_audit(code: GrossCode | None = None) -> dict:
    code = code or make_paper_gross_code()
    return {
        "source": "https://arxiv.org/html/2506.03094v1#A1",
        "equations_html": [31, 32, 34, 35],
        "paper_logical_order": list(PAPER_LOGICAL_ORDER),
        "pairing": (code.logical_x @ code.logical_z.T % 2).tolist(),
        "x_weights": code.logical_x.sum(axis=1).tolist(),
        "z_weights": code.logical_z.sum(axis=1).tolist(),
        "x_supports": [np.flatnonzero(v).tolist() for v in code.logical_x],
        "z_supports": [np.flatnonzero(v).tolist() for v in code.logical_z],
        "stabilizer_ranks": [rank(code.hx), rank(code.hz)],
        "X1_Z7_disjoint": not bool(np.any(code.logical_x[0] & code.logical_z[1])),
        "X1_X7_disjoint": not bool(np.any(code.logical_x[0] & code.logical_x[1])),
        "pivot_overlap_counts": (code.logical_x[:2] @ code.logical_z[:2].T).tolist(),
    }
