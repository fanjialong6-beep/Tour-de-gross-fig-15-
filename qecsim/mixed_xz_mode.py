from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import stim

from .gross_code import GrossCode, independent_rows, rank
from .lpu import MERGED, build_lpu
from .stim_circuit import BuiltCircuit, Check, CircuitReconstructor, GateTask, _edge_color
from .xz_paper_basis import make_paper_gross_code, paper_dual_index, paper_dual, paper_basis_audit


# Build a Stim Pauli string from binary X and Z supports.
def pauli_from_maps(n: int, x=(), z=()) -> stim.PauliString:
    a = np.zeros(n, dtype=np.bool_)
    b = np.zeros(n, dtype=np.bool_)
    a[list(x)] = True
    b[list(z)] = True
    return stim.PauliString.from_numpy(xs=a, zs=b)


# Represent a Pauli string as a binary symplectic row.
def symplectic_row(pauli: stim.PauliString) -> np.ndarray:
    x, z = pauli.to_numpy()
    return np.concatenate([x, z]).astype(np.uint8)


class MixedXZReconstructor(CircuitReconstructor):
    # Initialize the code, measurement geometry, and circuit state.
    def __init__(self, code: GrossCode | None = None, p: float = 0.0, target_sign: int = 1):
        if target_sign not in (-1, 1):
            raise ValueError("target_sign must be +1 or -1")
        self.target_sign = target_sign
        code = code or make_paper_gross_code()
        expected = make_paper_gross_code()
        if not (np.array_equal(code.logical_x, expected.logical_x)
                and np.array_equal(code.logical_z, expected.logical_z)):
            raise ValueError("XZ requires make_paper_gross_code() physical representatives")
        super().__init__(code, build_lpu(code, "X"), "X", p, schedule="graph_sync12")
        self.mode = "X1Z7_joint"
        self.schedule = "xz_phase_coloring"

    # Allocate reference, edge, and check qubits with their supports.
    def _allocate_qubits_and_checks(self) -> None:
        super()._allocate_qubits_and_checks()
        self.check_paulis: dict[str, dict[int, str]] = {
            check.name: {q: check.pauli for q in check.data} for check in self.checks
        }
        self.edge_x_check: dict[int, int] = {}
        for edge in self.lpu.edges:
            endpoints = [paper_dual_index(self.code, self.lpu.support[v][0])
                         for v in (edge.u, edge.v)]
            matches = [i for i, row in enumerate(self.code.hx)
                       if sorted(np.flatnonzero(row & self.code.logical_z[1]).tolist()) == sorted(endpoints)]
            if len(matches) != 1:
                raise AssertionError((edge.index, endpoints, matches))
            i = matches[0]
            if not np.array_equal(self.code.hx[i], paper_dual(self.code, self.code.hz[edge.z_check])):
                raise AssertionError("dual edge port is not the paper ZX-dual BB check")
            self.edge_x_check[edge.index] = i
            check = self.check_by_name[f"HX{i}"]
            q = self.edge_qubits[edge.index]
            check.data.append(q)
            check.partitions[0].append(q)
            check.pauli = "M"
            self.check_paulis[check.name][q] = "Z"
        for vertex in self.lpu.vertices:
            check = self.check_by_name[self.vertex_name[vertex]]
            q = self.lpu.support[vertex][0]
            dual_q = paper_dual_index(self.code, q)
            check.data.append(dual_q)
            check.partitions[1 if vertex == MERGED else 0].append(dual_q)
            self.check_paulis[check.name][dual_q] = "Z"
            check.pauli = "M"
        self.target = pauli_from_maps(self.code.n,
                                     np.flatnonzero(self.code.logical_x[0]),
                                     np.flatnonzero(self.code.logical_z[1]))
        self.preserved_pairs = [
            (pauli_from_maps(self.code.n, np.flatnonzero(self.code.logical_x[0])),
             pauli_from_maps(self.code.n, np.flatnonzero(self.code.logical_x[1]),
                             np.flatnonzero(self.code.logical_z[0])))
        ]
        self.preserved_pairs += [
            (pauli_from_maps(self.code.n, np.flatnonzero(self.code.logical_x[i])),
             pauli_from_maps(self.code.n, z=np.flatnonzero(self.code.logical_z[i])))
            for i in range(2, 12)
        ]

    # Prepare the measured logical eigenstate and reference Bell pairs.
    def _initial_state(self) -> None:
        n = self.code.n + len(self.ref_qubits)
        stabs = [pauli_from_maps(n, np.flatnonzero(r)) for r in independent_rows(self.code.hx)]
        stabs += [pauli_from_maps(n, z=np.flatnonzero(r)) for r in independent_rows(self.code.hz)]
        target = self.target.copy()
        target += stim.PauliString(n - self.code.n)
        target.sign = self.target_sign
        stabs.append(target)
        for (a, b), ref in zip(self.preserved_pairs, self.ref_qubits):
            for pauli, axis in ((a, "X"), (b, "Z")):
                pauli = pauli.copy()
                pauli += stim.PauliString(n - self.code.n)
                pauli[ref] = axis
                stabs.append(pauli)
        self.circuit += stim.Tableau.from_stabilizers(stabs).to_circuit("graph_state")

    # Construct the mixed-XZ schedule and compare it with the saved schedule.
    def _load_optimized_schedule(self):
        tasks = []
        merged = self.check_by_name[self.vertex_name[MERGED]]
        tasks.append(GateTask(1, "CX", *merged.ancillas, merged.name))
        for vertex in self.lpu.vertices:
            check = self.check_by_name[self.vertex_name[vertex]]
            for ancilla, partition in zip(check.ancillas, check.partitions):
                for offset, q in enumerate(q for q in partition if q < self.code.n):
                    tasks.append(GateTask(2 + offset, "C" + self.check_paulis[check.name][q],
                                          ancilla, q, check.name))
        offsets = {"HX": [(0,0),(0,1),(3,-1),(0,0),(1,0),(-1,-3)],
                   "HZ": [(0,0),(-1,0),(1,3),(0,0),(0,-1),(-3,1)]}
        for i in range(72):
            x, y = divmod(i, self.code.m)
            for kind, times in (("HX", self.X_TERM_TIMES), ("HZ", self.Z_TERM_TIMES)):
                check = self.check_by_name[f"{kind}{i}"]
                for k, (dx, dy) in enumerate(offsets[kind]):
                    q = self.code.index(0 if k < 3 else 1, x + dx, y + dy)
                    tasks.append(GateTask(times[k] + 3, "CX" if kind == "HX" else "CZ",
                                          check.ancillas[0], q, check.name))
        next_time = 11
        for family in ("vertex", "cycle"):
            pairs = []
            lookup = {}
            for check in self.checks:
                if check.family != family:
                    continue
                for ancilla, partition in zip(check.ancillas, check.partitions):
                    for q in partition:
                        if q in self.edge_qubits:
                            pair = (ancilla, q)
                            pairs.append(pair)
                            lookup[pair] = check.name
            colors = _edge_color(pairs)
            for offset, color in enumerate(colors):
                for ancilla, q in color:
                    name = lookup[(ancilla, q)]
                    tasks.append(GateTask(next_time + offset, "C" + self.check_paulis[name][q],
                                          ancilla, q, name))
            next_time += len(colors)
        for edge in self.lpu.edges:
            for offset, name in enumerate((f"HZ{edge.z_check}", f"HX{self.edge_x_check[edge.index]}")):
                tasks.append(GateTask(next_time + offset, "CZ", self.check_by_name[name].ancillas[0],
                                      self.edge_qubits[edge.index], name))
        tasks, bb_delta = self._asap_preserving_bb_layers(tasks)
        self.period = max(t.time for t in tasks) + 2
        intervals = {}
        for check in self.checks:
            for ancilla in check.ancillas:
                times = [t.time for t in tasks if ancilla in (t.a, t.b)]
                intervals[f"{check.name}:{ancilla}"] = (min(times) - 1, max(times) + 1)
        self.schedule_tasks = tasks
        self.schedule_intervals = intervals
        self.schedule_certificate = self._validate_general_schedule(tasks, intervals)
        self.schedule_payload = {
            "mode": self.mode, "schedule_method": "deterministic_phase_coloring_then_asap",
            "period": self.period, "bb_delta": bb_delta, "tasks": [asdict(t) for t in tasks],
            "intervals": intervals, "author_schedule": False,
        }
        frozen_path = Path(__file__).with_name("xz_schedule.json")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen != json.loads(json.dumps(self.schedule_payload)):
            raise ValueError("XZ schedule differs from frozen xz_schedule.json; use a new reviewed experiment")
        self.schedule_path = frozen_path
        return tasks, intervals

    # Construct the fixed XZ schedule while preserving relative BB gate layers.
    def _asap_preserving_bb_layers(self, catalog):
        by_qubit = defaultdict(list)
        phases = []
        ancillas = {a for check in self.checks for a in check.ancillas}
        for i, task in enumerate(catalog):
            family = self.check_by_name[task.check_name].family
            if task.b in ancillas:
                phase = (0, task.time)
            elif family == "vertex" and task.b < self.code.n:
                phase = (1, task.time)
            elif family in ("bb_x", "bb_z") and task.b < self.code.n:
                phase = (2, task.time)
            elif family == "vertex":
                phase = (3, task.time)
            elif family == "cycle":
                phase = (4, task.time)
            else:
                phase = (5, task.time)
            phases.append(phase)
            by_qubit[task.a].append(i)
            by_qubit[task.b].append(i)
        constraints = set()
        for indices in by_qubit.values():
            indices.sort(key=lambda i: (phases[i], i))
            constraints.update(zip(indices, indices[1:]))
        lookup = {(t.check_name, t.b): i for i, t in enumerate(catalog)}
        for first, second in combinations(self.checks, 2):
            a, b = self.check_paulis[first.name], self.check_paulis[second.name]
            for q in a.keys() & b.keys():
                if a[q] == b[q]:
                    continue
                i, j = lookup[first.name, q], lookup[second.name, q]
                constraints.add((i,j) if phases[i] < phases[j] else (j,i))
        variables = [0 if phase[0] == 2 else i+1 for i, phase in enumerate(phases)]
        offsets = [t.time-3 if phase[0] == 2 else 0 for t, phase in zip(catalog, phases)]
        values = [0] + [1]*len(catalog)
        constraints = sorted(constraints)
        for _ in range(len(catalog)+1):
            changed = False
            for i, j in constraints:
                vi, vj = variables[i], variables[j]
                required = values[vi] + offsets[i] + 1 - offsets[j]
                if values[vj] < required:
                    values[vj] = required
                    changed = True
            if not changed:
                break
        else:
            raise ValueError("inconsistent schedule precedence")
        tasks = [GateTask(values[variables[i]]+offsets[i], t.gate, t.a,t.b,t.check_name)
                 for i,t in enumerate(catalog)]
        differences = {tasks[i].time-offsets[i] for i,p in enumerate(phases) if p[0] == 2}
        assert differences == {values[0]}
        return tasks, values[0]

    # Validate mixed-Pauli check timing and repeated-round compatibility.
    def _validate_general_schedule(self, tasks, intervals) -> dict:
        errors = []
        occupied = {}
        overlap_cases = 0
        by_port = {(t.check_name, t.b): t.time for t in tasks if t.b in self.check_by_name[t.check_name].data}
        for cycle in range(self.CYCLES):
            offset = cycle * self.period
            events = [(offset + t.time, q, "gate") for t in tasks for q in (t.a, t.b)]
            for key, (start, end) in intervals.items():
                q = int(key.split(":")[1])
                events += [(offset + start, q, "reset"), (offset + end, q, "measure")]
                if end - start + 1 > self.period:
                    errors.append("ancilla lifetime exceeds period")
            for time, q, kind in events:
                if (time, q) in occupied:
                    errors.append(f"collision {time} {q}")
                occupied[time, q] = kind
        for c, other in combinations(self.checks, 2):
            a, b = self.check_paulis[c.name], self.check_paulis[other.name]
            overlap = [q for q in a.keys() & b.keys() if a[q] != b[q]]
            if len(overlap) % 2:
                errors.append(f"noncommuting checks {c.name} {other.name}")
            if not overlap:
                continue
            for r1 in range(self.CYCLES):
                for r2 in range(self.CYCLES):
                    signs = [by_port[c.name, q] - by_port[other.name, q] + (r1-r2)*self.period for q in overlap]
                    overlap_cases += 1
                    if any(v == 0 for v in signs) or sum(v < 0 for v in signs) % 2:
                        errors.append(f"invalid overlap order {c.name} {other.name}")
        bell_checks = [c for c in self.checks if len(c.ancillas) == 2]
        for check in bell_checks:
            bell = [t for t in tasks if t.check_name == check.name and t.b in check.ancillas]
            interactions = [t.time for t in tasks if t.check_name == check.name and t.b in check.data]
            if len(bell) != 1 or bell[0].time >= min(interactions):
                errors.append("Bell preparation not before interactions")
        if errors:
            raise AssertionError(errors[:10])
        return {"valid": True, "errors": [], "event_collision_count": 0,
                "general_pauli_overlap_pairs_checked": overlap_cases,
                "bell_checks": len(bell_checks), "bb_seven_layers_preserved": True,
                "repeat_period": self.period, "single_cycle_span": self.period,
                "cycles": self.CYCLES, "author_exact_coloring_verified": False}

    # Apply measurement-controlled Pauli corrections after edge readout.
    def _apply_gauge_correction(self, edge_records):
        for vertex, path in self.lpu.spanning_tree_paths(MERGED).items():
            if vertex == MERGED:
                continue
            name = self.vertex_name[vertex]
            for q, axis in self.check_paulis[name].items():
                if q >= self.code.n:
                    continue
                for edge in path:
                    self.circuit.append("C" + axis,
                                        [stim.target_rec(edge_records[edge] - self.measurement_count), q])

    # Find edge outcomes that anticommute with a final Pauli measurement.
    def _correction_records_for_pauli(self, pauli_map, edge_records):
        parity = np.zeros(len(self.edge_qubits), dtype=np.uint8)
        for vertex, path in self.lpu.spanning_tree_paths(MERGED).items():
            if vertex == MERGED:
                continue
            correction = self.check_paulis[self.vertex_name[vertex]]
            flips = sum(q < self.code.n and q in pauli_map and axis != pauli_map[q]
                        for q, axis in correction.items()) % 2
            if flips:
                parity[path] ^= 1
        return [edge_records[i] for i in np.flatnonzero(parity)]

    # Append an ideal mixed-Pauli product measurement.
    def _append_general_mpp(self, pauli: stim.PauliString) -> int:
        targets = []
        for q in range(len(pauli)):
            if pauli[q]:
                if targets:
                    targets.append(stim.target_combiner())
                targets.append({1: stim.target_x, 2: stim.target_y, 3: stim.target_z}[pauli[q]](q))
        self.circuit.append("MPP", targets)
        record = self.measurement_count
        self.measurement_count += 1
        return record

    # Build syndrome detectors and logical scoring observables.
    def _add_detectors_and_observables(self, edge_records, original_records):
        for check in self.checks:
            if check.family != "vertex":
                self._append_detector(self.records["deformed", 0, check.name])
        for cycle in range(1, self.CYCLES):
            for check in self.checks:
                self._append_detector(self.records["deformed", cycle-1, check.name]
                                      + self.records["deformed", cycle, check.name])
        for i in range(72):
            for kind in ("HX", "HZ"):
                attached = [e.index for e in self.lpu.edges
                            if (e.z_check == i if kind == "HZ" else self.edge_x_check[e.index] == i)]
                name = f"{kind}{i}"
                self._append_detector(self.records["deformed", self.CYCLES-1, name]
                                      + original_records[name] + [edge_records[e] for e in attached])
        for i, edges in enumerate(self.lpu.cycles):
            self._append_detector(self.records["deformed", self.CYCLES-1, f"U{i}"]
                                  + [edge_records[e] for e in edges])
        for axis, rows in (("X", self.code.hx), ("Z", self.code.hz)):
            for i, row in enumerate(rows):
                name = f"H{axis}{i}"
                qmap = {int(q): axis for q in np.flatnonzero(row)}
                final = self._append_mpp(axis, qmap, ("final", name))
                self._append_detector(original_records[name] + [final]
                                      + self._correction_records_for_pauli(qmap, edge_records))
        readout = sum((self.records["deformed", self.CYCLES-1, self.vertex_name[v]]
                       for v in self.lpu.vertices), [])
        self.readout_records = readout
        self._append_observable(0, readout)
        obs = 1
        for (a, b), ref in zip(self.preserved_pairs, self.ref_qubits):
            for pauli, axis in ((a, "X"), (b, "Z")):
                pauli = pauli.copy()
                pauli += stim.PauliString(self.code.n + len(self.ref_qubits) - len(pauli))
                pauli[ref] = axis
                self._append_observable(obs, [self._append_general_mpp(pauli)])
                obs += 1
        assert obs == 23

    # Verify mixed-Pauli check commutation, ranks, and logical generators.
    def structure_audit(self) -> dict:
        n = self.code.n + len(self.edge_qubits)
        edge_map = {q: self.code.n+i for i, q in enumerate(self.edge_qubits)}
        rows = []
        product = stim.PauliString(self.code.n)
        for check in self.checks:
            pauli = stim.PauliString(n)
            for q, axis in self.check_paulis[check.name].items():
                pauli[q if q < self.code.n else edge_map[q]] = axis
            rows.append(symplectic_row(pauli))
            if check.family == "vertex":
                small = stim.PauliString(self.code.n)
                for q, axis in self.check_paulis[check.name].items():
                    if q < self.code.n:
                        small[q] = axis
                product *= small
        mat = np.asarray(rows)
        commutator = (mat[:, :n] @ mat[:, n:].T + mat[:, n:] @ mat[:, :n].T) % 2
        generators = [symplectic_row(self.target)]
        for a, b in self.preserved_pairs:
            generators += [symplectic_row(a), symplectic_row(b)]
        old = np.block([[self.code.hx, np.zeros_like(self.code.hx)],
                        [np.zeros_like(self.code.hz), self.code.hz]])
        out = {"deformed_data_qubits": n, "deformed_stabilizer_rank": rank(mat),
               "remaining_logical_qubits": n-rank(mat), "commuting_stabilizers": not bool(commutator.any()),
               "target_weight": self.target.weight, "vertex_product_matches_target": product == self.target,
               "native_observables": len(generators), "native_generators_independent_mod_stabilizers":
               rank(np.vstack([old, generators]))-rank(old), "paper_basis": paper_basis_audit(self.code)}
        assert out["remaining_logical_qubits"] == 11
        assert out["commuting_stabilizers"] and out["vertex_product_matches_target"]
        assert out["native_generators_independent_mod_stabilizers"] == 23
        connectivity = defaultdict(set)
        for task in self.schedule_tasks:
            connectivity[task.a].add(task.b)
            connectivity[task.b].add(task.a)
        out["max_physical_connectivity_degree"] = max(map(len, connectivity.values()))
        out["lpu_physical_qubits"] = len(self.physical_qubits) - 288
        out["physical_port_proof"] = {
            "source": "arXiv:2506.03094v1 Appendix A.1 duality and A.3/A.4 shared half-LPU ports",
            "pi": "(b,x,y)->(1-b,1-x,1-y)",
            "vertex_ports": [{"vertex": repr(v), "X_port": int(self.lpu.support[v][0]),
                              "Z_port": paper_dual_index(self.code, self.lpu.support[v][0])}
                             for v in self.lpu.vertices],
            "edge_ports": [{"edge": e.index, "original_Z_check": e.z_check,
                            "dual_X_check": self.edge_x_check[e.index]} for e in self.lpu.edges],
            "all_dual_checks_match_full_BB_support": True,
            "LPU_internal_graph_and_cycles": "unchanged original left-half graph",
            "Bell_root_dual_port_on_opposite_existing_ancilla": True,
        }
        assert out["max_physical_connectivity_degree"] <= 7
        return out

    # Assemble the complete physical measurement circuit and its metadata.
    def build(self) -> BuiltCircuit:
        self._initial_state()
        self._edge_initialization()
        last_time, _ = self._populate_deformed_timeline()
        self._emit_timeline(0, last_time)
        edge_records = self._measure_edges()
        original_records, _ = self._append_noisy_original_cycle()
        self._apply_gauge_correction(edge_records)
        self._add_detectors_and_observables(edge_records, original_records)
        dem = self.circuit.detector_error_model(decompose_errors=False, flatten_loops=True,
                                               allow_gauge_detectors=False)
        assert dem.num_observables == 23
        metadata = {
            "schedule": self.schedule, "schedule_method": "deterministic_phase_coloring_then_asap",
            "schedule_certificate": self.schedule_certificate,
            "schedule_sha256": hashlib.sha256(json.dumps(self.schedule_payload, sort_keys=True).encode()).hexdigest(),
            "author_timestamps_available": False, "paper_baseline_verified": False,
            "new_experiment": True, "paper_direct_curve_available": False,
            "lpu_vertices": len(self.lpu.vertices), "lpu_edges": len(self.edge_qubits),
            "lpu_cycles": len(self.lpu.cycles), "lpu_physical_qubits": len(self.physical_qubits)-288,
            "lpu_edge_data_qubits": len(self.edge_qubits),
            "lpu_check_ancillas": sum(len(c.ancillas) for c in self.checks if c.family in ("vertex", "cycle")),
            "physical_qubits_excluding_ideal_references": len(self.physical_qubits),
            "physical_two_qubit_gates_per_deformed_cycle": len(self.schedule_tasks),
            "physical_two_qubit_gates_total": self.CYCLES*len(self.schedule_tasks) + 864,
            "deformed_cycles": self.CYCLES, "dem_errors": dem.num_errors,
            "total_noisy_instruction_ticks": 1 + last_time + 1 + 1 + 9,
            "native_observable_labels": ["X1Z7_readout", "X1", "Z1X7"]
                + [f"{axis}{i}" for i in (2,3,4,5,6,8,9,10,11,12) for axis in ("X","Z")],
            "physical_basis": "paper_A1_equations_31_to_35", "structure_audit": self.structure_audit(),
        }
        return BuiltCircuit(self.mode, self.p, self.circuit, self.fault_count, self.num_qubits,
                            dem.num_detectors, dem.num_observables, self.period, last_time+1, metadata)


# Build the joint X1Z7 physical measurement circuit.
def build_xz_circuit(p: float = 0.0, code: GrossCode | None = None) -> BuiltCircuit:
    return MixedXZReconstructor(code, p).build()
