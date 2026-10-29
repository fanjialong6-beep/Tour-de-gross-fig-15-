from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
import stim


# Return the SHA-256 digest of a byte sequence.
def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Set the physical noise rate while preserving every circuit operation.
def at_p(circuit: stim.Circuit, p: float) -> stim.Circuit:
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("physical p must be in [0,1]")
    text = re.sub(r"(?m)^((?:[XYZ]_ERROR|E|CORRELATED_ERROR))\([^)]*\)",
                  lambda match: f"{match[1]}({p / 15!r})", str(circuit.flattened()))
    return stim.Circuit(text)


# Compute odd-parity probabilities for groups of independent fault events.
def grouped_priors(multiplicity: np.ndarray, p: float) -> np.ndarray:
    if not np.isfinite(p) or not 0 < p <= 1:
        raise ValueError("decoder prior p must be in (0,1]")
    return -np.expm1(multiplicity * np.log1p(-2 * p / 15)) / 2


# Group identical detector and observable effects with exact multiplicities.
def dem_groups(circuit: stim.Circuit, q: float) -> dict:
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True,
                                       allow_gauge_detectors=False)
    groups = {}
    for instruction in dem:
        if instruction.type != "error":
            continue
        detectors, observables = [], []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                observables.append(target.val)
            else:
                raise ValueError("unexpected separator or decomposed DEM error")
        key = (tuple(detectors), tuple(observables))
        probability = instruction.args_copy()[0]
        real_m = np.log1p(-2 * probability) / np.log1p(-2 * q)
        m = int(round(real_m))
        if m < 1 or abs(m - real_m) > 1e-7:
            raise ValueError("DEM channel is not an integer number of equal-q independent faults")
        groups[key] = groups.get(key, 0) + m
    return groups


@dataclass
class FaultModel:
    h: sparse.csc_matrix
    a: sparse.csc_matrix
    multiplicity: np.ndarray
    circuit: stim.Circuit
    metadata: dict

    # Return the number of expanded nonsilent independent fault events.
    @property
    def n(self) -> int:
        return int(self.multiplicity.sum())

    # Compute syndrome and observable flips from odd fault-group indices.
    def effects(self, odd_groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(self.h[:, odd_groups].sum(axis=1)).ravel().astype(np.uint64) % 2
        o = np.asarray(self.a[:, odd_groups].sum(axis=1)).ravel().astype(np.uint64) % 2
        return d.astype(np.uint8), o.astype(np.uint8)

    # Sample distinct expanded fault events and reduce their group parities.
    def sample_weight(self, weight: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= weight <= self.n:
            raise ValueError("weight must lie between 0 and effective independent fault count")
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(self.n, size=weight, replace=False))
        groups = np.searchsorted(np.cumsum(self.multiplicity), ids, side="right")
        unique, counts = np.unique(groups, return_counts=True)
        d, o = self.effects(unique[counts % 2 == 1])
        return d, o, ids

    # Write sparse matrices, multiplicities, circuit, and model metadata.
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "H.npz", self.h)
        sparse.save_npz(directory / "A.npz", self.a)
        np.save(directory / "multiplicity.npy", self.multiplicity)
        self.circuit.to_file(str(directory / "reference.stim"))
        (directory / "model.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    # Read a saved fault model and verify its scientific content hash.
    @classmethod
    def load(cls, directory: Path) -> "FaultModel":
        model = cls(sparse.load_npz(directory / "H.npz").tocsc(),
                    sparse.load_npz(directory / "A.npz").tocsc(),
                    np.load(directory / "multiplicity.npy"),
                    stim.Circuit.from_file(str(directory / "reference.stim")),
                    json.loads((directory / "model.json").read_text(encoding="utf-8")))
        if model.content_hash() != model.metadata["model_sha256"]:
            raise ValueError("Model files changed; refusing to resume")
        return model

    # Hash the canonical circuit, sparse matrices, and fault multiplicities.
    def content_hash(self) -> str:
        chunks = [str(self.circuit).encode(), self.multiplicity.astype("<i8").tobytes()]
        for matrix in (self.h, self.a):
            chunks += [np.asarray(matrix.shape, dtype="<i8").tobytes(),
                       matrix.indptr.astype("<i8").tobytes(),
                       matrix.indices.astype("<i8").tobytes(),
                       matrix.data.astype(np.uint8).tobytes()]
        return digest(b"".join(chunks))


# Build exact grouped H and A matrices from a physical Stim circuit.
def model_from_circuit(circuit: stim.Circuit, raw_count: int, metadata: dict | None = None) -> FaultModel:
    circuit = at_p(circuit, .003)
    groups = dem_groups(circuit, .003 / 15)
    if groups != dem_groups(at_p(circuit, .001), .001 / 15):
        raise ValueError("Fault multiplicities depend on probe probability")
    keys = sorted(groups)
    multiplicity = np.asarray([groups[k] for k in keys], dtype=np.int64)
    matrices = []
    for component, rows in [(0, circuit.num_detectors), (1, circuit.num_observables)]:
        indices = [r for key in keys for r in key[component]]
        indptr = np.r_[0, np.cumsum([len(key[component]) for key in keys])]
        matrices.append(sparse.csc_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr),
                                          shape=(rows, len(keys))))
    effective = int(multiplicity.sum())
    if effective > raw_count:
        raise ValueError("DEM has more independent faults than circuit ledger")
    same_h = {}
    for d, o in keys:
        same_h.setdefault(d, set()).add(o)
    meta = {**(metadata or {}), "raw_fault_count": raw_count,
            "effective_fault_count": effective, "silent_fault_count": raw_count - effective,
            "compressed_columns": len(keys), "weight_convention": "non_silent_independent_q_faults",
            "multiplicity_probes_p": [.001, .003],
            "same_syndrome_different_logical_groups": sum(len(v) > 1 for v in same_h.values()),
            "pure_logical_groups": sum(not d and bool(o) for d, o in keys),
            "reference_p": .003, "reference_circuit_sha256": digest(str(circuit).encode())}
    model = FaultModel(matrices[0], matrices[1], multiplicity, circuit, meta)
    meta["model_sha256"] = model.content_hash()
    return model
