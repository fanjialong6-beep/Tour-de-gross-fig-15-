from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
from typing import Hashable, Iterable

import numpy as np
import stim

from .gross_code import GrossCode, independent_rows, measurement_basis
from .lpu import LpuGraph, MERGED


SCHEDULE_PREFIXES = {"graph_sync12": "sync12"}


# Select the frozen schedule resource for a measurement.
def schedule_filename(schedule: str, mode: str) -> str:
    if schedule not in SCHEDULE_PREFIXES or mode not in {"X", "XX"}:
        raise ValueError(f"unknown schedule or measurement mode: {schedule}, {mode}")
    return f"{SCHEDULE_PREFIXES[schedule]}_{mode}.json"


@dataclass
class Check:
    name: str
    pauli: str
    data: list[int]
    ancillas: list[int]
    partitions: list[list[int]]
    family: str


@dataclass(frozen=True)
class GateTask:
    time: int
    gate: str
    a: int
    b: int
    check_name: str


@dataclass
class BuiltCircuit:
    mode: str
    p: float
    circuit: stim.Circuit
    fault_count: int
    num_qubits: int
    num_detectors: int
    num_observables: int
    schedule_period: int
    schedule_span: int
    metadata: dict = field(default_factory=dict)


# Partition a bipartite coupling graph into disjoint gate layers.
def _edge_color(edges: list[tuple[Hashable, Hashable]]) -> list[list[tuple[Hashable, Hashable]]]:
    if not edges:
        return []
    left_values = list(dict.fromkeys(a for a, _ in edges))
    right_values = list(dict.fromkeys(b for _, b in edges))
    li = {v: i for i, v in enumerate(left_values)}
    ri = {v: i for i, v in enumerate(right_values)}
    nl, nr = len(left_values), len(right_values)
    degree_l = [0] * nl
    degree_r = [0] * nr
    buckets: dict[tuple[int, int], list[tuple[Hashable, Hashable] | None]] = defaultdict(list)
    for edge in edges:
        a, b = li[edge[0]], ri[edge[1]]
        degree_l[a] += 1
        degree_r[b] += 1
        buckets[(a, b)].append(edge)
    delta = max(max(degree_l), max(degree_r))
    n = max(nl, nr)
    degree_l += [0] * (n - nl)
    degree_r += [0] * (n - nr)
    missing_l = [i for i, d in enumerate(degree_l) for _ in range(delta - d)]
    missing_r = [i for i, d in enumerate(degree_r) for _ in range(delta - d)]
    for a, b in zip(missing_l, missing_r):
        buckets[(a, b)].append(None)

    colors: list[list[tuple[Hashable, Hashable]]] = []
    for _ in range(delta):
        adjacency = [[] for _ in range(n)]
        for (a, b), payloads in buckets.items():
            if payloads:
                adjacency[a].append(b)
        for row in adjacency:
            row.sort()
        pair_l = [-1] * n
        pair_r = [-1] * n

        # Extend a matching along an available alternating path.
        def augment(a: int, seen: set[int]) -> bool:
            for b in adjacency[a]:
                if b in seen:
                    continue
                seen.add(b)
                if pair_r[b] == -1 or augment(pair_r[b], seen):
                    pair_l[a] = b
                    pair_r[b] = a
                    return True
            return False

        for a in range(n):
            if not augment(a, set()):
                raise RuntimeError("failed to edge-color bipartite graph")
        color: list[tuple[Hashable, Hashable]] = []
        for a, b in enumerate(pair_l):
            payload = buckets[(a, b)].pop()
            if not buckets[(a, b)]:
                del buckets[(a, b)]
            if payload is not None:
                color.append(payload)
        colors.append(color)
    return colors


# Build a Stim Pauli string from X and Z supports.
def _pauli_string(n: int, x: Iterable[int] = (), z: Iterable[int] = ()) -> stim.PauliString:
    result = stim.PauliString(n)
    for q in x:
        result[q] = 1
    for q in z:
        result[q] = 3
    return result


# Convert a Pauli support into Stim product-measurement targets.
def _mpp_targets(pauli: str, qubits: Iterable[int]) -> list[stim.GateTarget]:
    targets: list[stim.GateTarget] = []
    qubits = list(qubits)
    for i, q in enumerate(qubits):
        if i:
            targets.append(stim.target_combiner())
        targets.append(stim.target_x(q) if pauli == "X" else stim.target_z(q))
    return targets


class CircuitReconstructor:
    CYCLES = 10

    X_TERM_TIMES = [2, 7, 6, 3, 5, 4]
    Z_TERM_TIMES = [4, 5, 3, 6, 2, 1]

    # Initialize the code, measurement geometry, and circuit state.
    def __init__(self, code: GrossCode, lpu: LpuGraph, mode: str, p: float,
                 schedule: str = "graph_sync12"):
        if schedule not in SCHEDULE_PREFIXES:
            raise ValueError(f"schedule must be one of {tuple(SCHEDULE_PREFIXES)}")
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be finite and between 0 and 1")
        if mode not in {"X", "XX"}:
            raise ValueError("mode must be X or XX")
        schedule_filename(schedule, mode)
        self.schedule = schedule
        self.code = code
        self.lpu = lpu
        self.mode = mode
        self.p = float(p)
        self.q = self.p / 15
        self.circuit = stim.Circuit()
        self.fault_count = 0
        self.measurement_count = 0
        self.period = 0
        self.records: dict[tuple, list[int]] = {}
        self.timeline: dict[int, list[tuple]] = defaultdict(list)
        self.active_intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._allocate_qubits_and_checks()

    # Allocate reference, edge, and check qubits with their supports.
    def _allocate_qubits_and_checks(self) -> None:
        n = self.code.n
        self.ref_qubits = list(range(n, n + 11))
        cursor = n + 11
        self.edge_qubits = list(range(cursor, cursor + len(self.lpu.edges)))
        cursor += len(self.edge_qubits)

        self.checks: list[Check] = []
        self.check_by_name: dict[str, Check] = {}

        # Allocate ancillas and register one parity check.
        def add_check(name: str, pauli: str, data: list[int], family: str,
                      partitions: list[list[int]] | None = None) -> Check:
            nonlocal cursor
            partitions = partitions or [list(data)]
            ancillas = list(range(cursor, cursor + len(partitions)))
            cursor += len(partitions)
            check = Check(name, pauli, list(data), ancillas, partitions, family)
            self.checks.append(check)
            self.check_by_name[name] = check
            return check

        for i, row in enumerate(self.code.hx):
            add_check(f"HX{i}", "X", np.flatnonzero(row).tolist(), "bb_x")
        for i, row in enumerate(self.code.hz):
            data = np.flatnonzero(row).tolist()
            data += [self.edge_qubits[e.index] for e in self.lpu.edges if e.z_check == i]
            add_check(f"HZ{i}", "Z", data, "bb_z")

        for v_index, vertex in enumerate(self.lpu.vertices):
            edge_data = [self.edge_qubits[e] for e in self.lpu.incident_edges(vertex)]
            code_data = list(self.lpu.support[vertex])
            data = code_data + edge_data
            if vertex == MERGED:
                left_code = [q for q in code_data if q in np.flatnonzero(self.code.logical_x[0])]
                right_code = [q for q in code_data if q in np.flatnonzero(self.code.logical_x[1])]
                left_edges = [self.edge_qubits[e.index] for e in self.lpu.edges
                              if e.kind == "left" and vertex in {e.u, e.v}]
                right_edges = [self.edge_qubits[e.index] for e in self.lpu.edges
                               if e.kind == "right" and vertex in {e.u, e.v}]
                partitions = [left_code + left_edges, right_code + right_edges]
            else:
                partitions = [data]
            add_check(f"V{v_index}", "X", data, "vertex", partitions)
            self.vertex_name = getattr(self, "vertex_name", {})
            self.vertex_name[vertex] = f"V{v_index}"

        for cycle_index, edge_indices in enumerate(self.lpu.cycles):
            add_check(
                f"U{cycle_index}",
                "Z",
                [self.edge_qubits[e] for e in edge_indices],
                "cycle",
            )

        self.num_qubits = cursor
        self.physical_qubits = set(range(self.code.n)) | set(self.edge_qubits)
        self.physical_qubits |= {a for check in self.checks for a in check.ancillas}

    # Prepare the measured logical eigenstate and reference Bell pairs.
    def _initial_state(self) -> None:
        x_basis, z_basis = measurement_basis(self.code, self.mode)
        n_state = self.code.n + len(self.ref_qubits)
        stabilizers: list[stim.PauliString] = []
        for row in independent_rows(self.code.hx):
            stabilizers.append(_pauli_string(n_state, x=np.flatnonzero(row)))
        for row in independent_rows(self.code.hz):
            stabilizers.append(_pauli_string(n_state, z=np.flatnonzero(row)))
        stabilizers.append(_pauli_string(n_state, x=np.flatnonzero(x_basis[0])))
        for i in range(1, 12):
            ref = self.ref_qubits[i - 1]
            stabilizers.append(_pauli_string(n_state, x=[*np.flatnonzero(x_basis[i]), ref]))
            stabilizers.append(_pauli_string(n_state, z=[*np.flatnonzero(z_basis[i]), ref]))
        if len(stabilizers) != n_state:
            raise AssertionError((len(stabilizers), n_state))
        tableau = stim.Tableau.from_stabilizers(stabilizers)
        self.circuit += tableau.to_circuit("graph_state")

    # Append independent single-qubit Pauli fault events.
    def _fault_1q(self, pauli: str, qubit: int, repeats: int) -> None:
        if self.q:
            self.circuit.append(f"{pauli}_ERROR", [qubit] * repeats, self.q)
        self.fault_count += repeats

    # Append the idle depolarizing fault events for one qubit.
    def _fault_idle(self, qubit: int) -> None:
        self._fault_1q("X", qubit, 5)
        self._fault_1q("Y", qubit, 5)
        self._fault_1q("Z", qubit, 5)

    # Append the fifteen independent two-qubit Pauli fault events.
    def _fault_2q(self, a: int, b: int) -> None:
        paulis = "IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ".split()
        if self.q:
            for pair in paulis:
                targets = []
                if pair[0] != "I":
                    targets.append({"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}[pair[0]](a))
                if pair[1] != "I":
                    targets.append({"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}[pair[1]](b))
                self.circuit.append("E", targets, self.q)
        self.fault_count += 15

    # Append noisy measurements and retain their record indices.
    def _record_measurements(self, keys: list[tuple], qubits: list[int], basis: str) -> None:
        for q in qubits:
            self._fault_1q("Z" if basis == "X" else "X", q, 15)
        self.circuit.append("MX" if basis == "X" else "M", qubits)
        for key in keys:
            self.records.setdefault(key, []).append(self.measurement_count)
            self.measurement_count += 1

    # Construct the gate catalog and check-ancilla lifetimes.
    def _schedule_template(self) -> tuple[list[GateTask], dict[str, tuple[int, int]]]:
        tasks: list[GateTask] = []
        shift = 3

        x_offsets = [(0, 0), (0, 1), (3, -1), (0, 0), (1, 0), (-1, -3)]
        z_offsets = [(0, 0), (-1, 0), (1, 3), (0, 0), (0, -1), (-3, 1)]
        for check_index in range(72):
            x, y = divmod(check_index, 6)
            x_check = self.check_by_name[f"HX{check_index}"]
            z_check = self.check_by_name[f"HZ{check_index}"]
            x_targets = [
                self.code.index(0 if k < 3 else 1, x + dx, y + dy)
                for k, (dx, dy) in enumerate(x_offsets)
            ]
            z_targets = [
                self.code.index(0 if k < 3 else 1, x + dx, y + dy)
                for k, (dx, dy) in enumerate(z_offsets)
            ]
            for k, data in enumerate(x_targets):
                tasks.append(GateTask(self.X_TERM_TIMES[k] + shift, "CX", x_check.ancillas[0], data, x_check.name))
            for k, data in enumerate(z_targets):
                tasks.append(GateTask(self.Z_TERM_TIMES[k] + shift, "CZ", z_check.ancillas[0], data, z_check.name))

        merged = self.check_by_name[self.vertex_name[MERGED]]
        tasks.append(GateTask(1, "CX", merged.ancillas[0], merged.ancillas[1], merged.name))

        for vertex in self.lpu.vertices:
            check = self.check_by_name[self.vertex_name[vertex]]
            for partition, ancilla in zip(check.partitions, check.ancillas):
                for data in partition:
                    if data < self.code.n:
                        tasks.append(GateTask(2, "CX", ancilla, data, check.name))

        vertex_edges: list[tuple[str, int]] = []
        vertex_task_lookup: dict[tuple[str, int], tuple[int, int]] = {}
        for vertex in self.lpu.vertices:
            check = self.check_by_name[self.vertex_name[vertex]]
            for partition, ancilla in zip(check.partitions, check.ancillas):
                for data in partition:
                    if data in self.edge_qubits:
                        pair = (check.name, data)
                        vertex_edges.append(pair)
                        vertex_task_lookup[pair] = (ancilla, data)
        vertex_colors = _edge_color(vertex_edges)
        for color_index, color in enumerate(vertex_colors):
            for pair in color:
                ancilla, data = vertex_task_lookup[pair]
                tasks.append(GateTask(3 + color_index, "CX", ancilla, data, pair[0]))

        cycle_edges: list[tuple[str, int]] = []
        cycle_task_lookup: dict[tuple[str, int], tuple[int, int]] = {}
        for check in self.checks:
            if check.family != "cycle":
                continue
            for data in check.data:
                pair = (check.name, data)
                cycle_edges.append(pair)
                cycle_task_lookup[pair] = (check.ancillas[0], data)
        cycle_colors = _edge_color(cycle_edges)
        cycle_start = 3 + len(vertex_colors)
        for color_index, color in enumerate(cycle_colors):
            for pair in color:
                ancilla, data = cycle_task_lookup[pair]
                tasks.append(GateTask(cycle_start + color_index, "CZ", ancilla, data, pair[0]))

        extra_time = cycle_start + len(cycle_colors)
        for edge in self.lpu.edges:
            if edge.z_check is not None:
                check = self.check_by_name[f"HZ{edge.z_check}"]
                tasks.append(GateTask(extra_time, "CZ", check.ancillas[0], self.edge_qubits[edge.index], check.name))

        self.period = extra_time + 2

        intervals: dict[str, tuple[int, int]] = {}
        for check in self.checks:
            check_tasks = [task for task in tasks if task.check_name == check.name]
            for ancilla in check.ancillas:
                times = [task.time for task in check_tasks if ancilla in {task.a, task.b}]
                if not times:
                    times = [1]
                key = f"{check.name}:{ancilla}"
                intervals[key] = (min(times) - 1, max(times) + 1)

        self._validate_template(tasks, intervals)
        return tasks, intervals

    # Check gate collisions and parity-check overlap order.
    def _validate_template(self, tasks: list[GateTask], intervals: dict[str, tuple[int, int]]) -> None:
        occupied: dict[tuple[int, int], GateTask] = {}
        for task in tasks:
            for q in (task.a, task.b):
                key = (task.time, q)
                if key in occupied:
                    raise AssertionError(f"schedule collision: {task} and {occupied[key]}")
                occupied[key] = task

        by_check_data = {(task.check_name, task.b): task.time for task in tasks
                         if task.gate in {"CX", "CZ"} and task.b < self.num_qubits}
        x_checks = [c for c in self.checks if c.pauli == "X"]
        z_checks = [c for c in self.checks if c.pauli == "Z"]
        for xc in x_checks:
            for zc in z_checks:
                overlap = sorted(set(xc.data) & set(zc.data))
                if not overlap:
                    continue
                if len(overlap) != 2:
                    raise AssertionError(f"non-commuting reconstructed checks: {xc.name}, {zc.name}, {overlap}")
                differences = [by_check_data[(xc.name, q)] - by_check_data[(zc.name, q)] for q in overlap]
                if not (all(d < 0 for d in differences) or all(d > 0 for d in differences)):
                    raise AssertionError(f"invalid overlap order: {xc.name}, {zc.name}, {differences}")

        for q in self.physical_qubits:
            times = sorted(task.time for task in tasks if q in {task.a, task.b})
            if times and times[-1] - times[0] >= self.period:
                raise AssertionError(f"period too short for data qubit {q}: {times}")
        for check in self.checks:
            for ancilla in check.ancillas:
                start, end = intervals[f"{check.name}:{ancilla}"]
                if end - start + 1 > self.period:
                    raise AssertionError(f"period too short for check ancilla {ancilla}")

    # Place ten repeated check rounds on one physical timeline.
    def _populate_deformed_timeline(self) -> tuple[int, dict[str, tuple[int, int]]]:
        tasks, template_intervals = self._load_optimized_schedule()
        for cycle in range(self.CYCLES):
            offset = cycle * self.period
            for task in tasks:
                self.timeline[offset + task.time].append(("gate", task.gate, task.a, task.b))
            for check in self.checks:
                for ancilla_index, ancilla in enumerate(check.ancillas):
                    start, end = template_intervals[f"{check.name}:{ancilla}"]
                    start += offset
                    end += offset
                    self.active_intervals[ancilla].append((start, end))
                    if len(check.ancillas) == 2:
                        basis = "X" if ancilla_index == 0 else "Z"
                    else:
                        basis = "X"
                    self.timeline[start].append(("reset", basis, ancilla))
                    self.timeline[end].append(("measure", "X", ancilla, ("deformed", cycle, check.name)))
        return max(self.timeline), template_intervals

    # Load the X or XX gate schedule and validate its repetition period.
    def _load_optimized_schedule(self) -> tuple[list[GateTask], dict[str, tuple[int, int]]]:
        path = Path(__file__).with_name("schedules") / schedule_filename(self.schedule, self.mode)
        if not path.exists():
            raise FileNotFoundError(
                f"missing frozen schedule {path}; restore the matching packaged schedule"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != self.mode:
            raise ValueError("Schedule measurement mode does not match the requested mode")
        tasks = [GateTask(**item) for item in payload["tasks"]]
        if (payload.get("seed") != {"X": 0, "XX": 13}[self.mode]
                or payload.get("interpretation") != "locked_layers_and_phases_plus_commuting_CZ_ASAP"
                or payload.get("hypothesis_not_author_confirmed") is not True):
            raise ValueError("Unexpected frozen graph_sync12 candidate metadata")
        selected_period = int(payload["span"])
        delta = 2
        self.period = selected_period
        if self.period != 12:
            raise ValueError("Frozen schedule must have a 12-step single-cycle span")
        self.schedule_path = path
        self.schedule_payload = payload
        offsets = {
            "bb_x": [(0, 0), (0, 1), (3, -1), (0, 0), (1, 0), (-1, -3)],
            "bb_z": [(0, 0), (-1, 0), (1, 3), (0, 0), (0, -1), (-3, 1)],
        }
        term_times = {"bb_x": self.X_TERM_TIMES, "bb_z": self.Z_TERM_TIMES}
        actual = {(t.check_name, t.b): t.time for t in tasks if t.b < self.code.n}
        for check in self.checks:
            if check.family not in offsets:
                continue
            x, y = divmod(int(check.name[2:]), self.code.m)
            for k, (dx, dy) in enumerate(offsets[check.family]):
                q = self.code.index(0 if k < 3 else 1, x + dx, y + dy)
                if actual.get((check.name, q)) != term_times[check.family][k] + delta:
                    raise ValueError("Stale or incompatible BB schedule; regenerate schedules")
        from collections import Counter
        expected, _ = self._schedule_template()
        gate_key = lambda t: (t.gate, t.a, t.b, t.check_name)
        if Counter(map(gate_key, tasks)) != Counter(map(gate_key, expected)):
            raise ValueError("Schedule gate set differs from the circuit connectivity")
        self.period = selected_period
        intervals: dict[str, tuple[int, int]] = {}
        for check in self.checks:
            check_tasks = [task for task in tasks if task.check_name == check.name]
            for ancilla in check.ancillas:
                times = [task.time for task in check_tasks if ancilla in {task.a, task.b}]
                if not times:
                    raise AssertionError(f"check ancilla {ancilla} has no scheduled gates")
                intervals[f"{check.name}:{ancilla}"] = (min(times) - 1, max(times) + 1)
        self._validate_template(tasks, intervals)
        if min(start for start, end in intervals.values()) != 0 or max(end for start, end in intervals.values()) != 11:
            raise ValueError("Graph single-cycle template must occupy exactly timesteps 0..11")
        for check in self.checks:
            if len(check.ancillas) == 2:
                bell = [t for t in tasks if t.a in check.ancillas and t.b in check.ancillas]
                interactions = [t.time for t in tasks if t.check_name == check.name and t.b in check.data]
                if len(bell) != 1 or bell[0].time >= min(interactions):
                    raise ValueError("Invalid Bell preparation order")
        from .schedule_certificate import certify_schedule
        self.schedule_certificate = certify_schedule(tasks, self.checks, self.period, cycles=self.CYCLES)
        if not self.schedule_certificate["valid"]:
            raise ValueError(f"Invalid repeated schedule: {self.schedule_certificate}")
        return tasks, intervals

    # Determine whether a qubit accumulates idle noise at this time.
    def _is_active(self, qubit: int, time: int) -> bool:
        if qubit < self.code.n or qubit in self.edge_qubits:
            return True
        return any(start <= time <= end for start, end in self.active_intervals.get(qubit, []))

    # Emit scheduled operations and idle noise as a Stim circuit.
    def _emit_timeline(self, first_time: int, last_time: int) -> None:
        for time in range(first_time, last_time + 1):
            events = self.timeline.get(time, [])
            occupied: set[int] = set()
            resets_x: list[int] = []
            resets_z: list[int] = []
            gates_cx: list[tuple[int, int]] = []
            gates_cz: list[tuple[int, int]] = []
            measures: list[tuple[int, tuple]] = []
            for event in events:
                if event[0] == "reset":
                    _, basis, q = event
                    (resets_x if basis == "X" else resets_z).append(q)
                    if q in occupied:
                        raise AssertionError((time, q, event))
                    occupied.add(q)
                elif event[0] == "gate":
                    _, gate, a, b = event
                    if a in occupied or b in occupied:
                        raise AssertionError((time, event))
                    occupied |= {a, b}
                    (gates_cx if gate == "CX" else gates_cz).append((a, b))
                elif event[0] == "measure":
                    _, basis, q, key = event
                    if q in occupied:
                        raise AssertionError((time, q, event))
                    occupied.add(q)
                    measures.append((q, key))

            if resets_x:
                self.circuit.append("RX", resets_x)
                for q in resets_x:
                    self._fault_1q("Z", q, 15)
            if resets_z:
                self.circuit.append("R", resets_z)
                for q in resets_z:
                    self._fault_1q("X", q, 15)
            for pairs, gate in ((gates_cx, "CX"), (gates_cz, "CZ")):
                if pairs:
                    self.circuit.append(gate, [q for pair in pairs for q in pair])
                    for a, b in pairs:
                        self._fault_2q(a, b)
            if measures:
                self._record_measurements(
                    [key for _, key in measures],
                    [q for q, _ in measures],
                    "X",
                )

            for q in sorted(self.physical_qubits - occupied):
                if self._is_active(q, time):
                    self._fault_idle(q)
            self.circuit.append("TICK")

    # Convert absolute measurement indices into relative Stim targets.
    def _rec_targets(self, absolute_records: Iterable[int]) -> list[stim.GateTarget]:
        return [stim.target_rec(record - self.measurement_count) for record in absolute_records]

    # Append one parity detector from measurement records.
    def _append_detector(self, absolute_records: Iterable[int]) -> None:
        self.circuit.append("DETECTOR", self._rec_targets(absolute_records))

    # Append an observable parity from measurement records.
    def _append_observable(self, index: int, absolute_records: Iterable[int]) -> None:
        self.circuit.append("OBSERVABLE_INCLUDE", self._rec_targets(absolute_records), index)

    # Append an ideal Pauli-product measurement and record its index.
    def _append_mpp(self, pauli: str, qubits: Iterable[int], key: tuple) -> int:
        self.circuit.append("MPP", _mpp_targets(pauli, qubits))
        record = self.measurement_count
        self.records[key] = [record]
        self.measurement_count += 1
        return record

    # Initialize LPU edge qubits with preparation and data-idle noise.
    def _edge_initialization(self) -> None:
        self.circuit.append("R", self.edge_qubits)
        for q in self.edge_qubits:
            self._fault_1q("X", q, 15)
        for q in range(self.code.n):
            self._fault_idle(q)
        self.circuit.append("TICK")

    # Append the noisy return cycle of original BB-code checks.
    def _append_noisy_original_cycle(self) -> tuple[dict[str, list[int]], int]:
        original_records: dict[str, list[int]] = {}
        start = 0
        local_events: dict[int, list[tuple]] = defaultdict(list)
        for i in range(72):
            x_check = self.check_by_name[f"HX{i}"]
            z_check = self.check_by_name[f"HZ{i}"]
            local_events[start].append(("reset", "X", z_check.ancillas[0]))
            local_events[start + 1].append(("reset", "X", x_check.ancillas[0]))
            local_events[start + 7].append(("measure", "X", z_check.ancillas[0], ("original", f"HZ{i}")))
            local_events[start + 8].append(("measure", "X", x_check.ancillas[0], ("original", f"HX{i}")))

            x, y = divmod(i, 6)
            x_offsets = [(0, 0), (0, 1), (3, -1), (0, 0), (1, 0), (-1, -3)]
            z_offsets = [(0, 0), (-1, 0), (1, 3), (0, 0), (0, -1), (-3, 1)]
            for k, (dx, dy) in enumerate(x_offsets):
                data = self.code.index(0 if k < 3 else 1, x + dx, y + dy)
                local_events[start + self.X_TERM_TIMES[k]].append(("gate", "CX", x_check.ancillas[0], data))
            for k, (dx, dy) in enumerate(z_offsets):
                data = self.code.index(0 if k < 3 else 1, x + dx, y + dy)
                local_events[start + self.Z_TERM_TIMES[k]].append(("gate", "CZ", z_check.ancillas[0], data))

        data_qubits = set(range(self.code.n))
        x_ancillas = {self.check_by_name[f"HX{i}"].ancillas[0] for i in range(72)}
        z_ancillas = {self.check_by_name[f"HZ{i}"].ancillas[0] for i in range(72)}
        for time in range(9):
            occupied: set[int] = set()
            events = local_events[time]
            resets = [e[2] for e in events if e[0] == "reset"]
            if resets:
                self.circuit.append("RX", resets)
                for q in resets:
                    occupied.add(q)
                    self._fault_1q("Z", q, 15)
            for gate in ("CX", "CZ"):
                pairs = [(e[2], e[3]) for e in events if e[0] == "gate" and e[1] == gate]
                if pairs:
                    self.circuit.append(gate, [q for pair in pairs for q in pair])
                    for a, b in pairs:
                        if a in occupied or b in occupied:
                            raise AssertionError((time, a, b))
                        occupied |= {a, b}
                        self._fault_2q(a, b)
            measurements = [(e[2], e[3]) for e in events if e[0] == "measure"]
            if measurements:
                self._record_measurements(
                    [key for _, key in measurements],
                    [q for q, _ in measurements],
                    "X",
                )
                occupied |= {q for q, _ in measurements}
            active = data_qubits | (z_ancillas if time <= 7 else set())
            active |= x_ancillas if time >= 1 else set()
            for q in sorted(active - occupied):
                self._fault_idle(q)
            self.circuit.append("TICK")
        for i in range(72):
            original_records[f"HX{i}"] = self.records[("original", f"HX{i}")]
            original_records[f"HZ{i}"] = self.records[("original", f"HZ{i}")]
        return original_records, 9

    # Measure the LPU edge qubits and account for data idle noise.
    def _measure_edges(self) -> list[int]:
        self._record_measurements(
            [("edge", i) for i in range(len(self.edge_qubits))],
            self.edge_qubits,
            "Z",
        )
        edge_records = [self.records[("edge", i)][0] for i in range(len(self.edge_qubits))]
        for q in range(self.code.n):
            self._fault_idle(q)
        self.circuit.append("TICK")

        return edge_records

    # Apply measurement-controlled Pauli corrections after edge readout.
    def _apply_gauge_correction(self, edge_records: list[int]) -> None:
        paths = self.lpu.spanning_tree_paths(MERGED)
        for vertex, path in paths.items():
            if vertex == MERGED:
                continue
            targets: list[stim.GateTarget | int] = []
            for edge_index in path:
                targets.extend([stim.target_rec(edge_records[edge_index] - self.measurement_count), self.lpu.support[vertex][0]])
            if targets:
                self.circuit.append("CX", targets)

    # Find edge outcomes that flip a later Z-type measurement.
    def _correction_flip_records(self, z_support: Iterable[int], edge_records: list[int]) -> list[int]:
        z_support = set(z_support)
        parity = np.zeros(len(self.lpu.edges), dtype=np.uint8)
        paths = self.lpu.spanning_tree_paths(MERGED)
        for vertex, path in paths.items():
            if vertex == MERGED:
                continue
            if self.lpu.support[vertex][0] in z_support:
                for edge_index in path:
                    parity[edge_index] ^= 1
        return [edge_records[i] for i in np.flatnonzero(parity)]

    # Build syndrome detectors and logical scoring observables.
    def _add_detectors_and_observables(self, edge_records: list[int], original_records: dict[str, list[int]]) -> None:
        for check in self.checks:
            if check.family != "vertex":
                self._append_detector(self.records[("deformed", 0, check.name)])
        for cycle in range(1, self.CYCLES):
            for check in self.checks:
                self._append_detector(
                    self.records[("deformed", cycle - 1, check.name)]
                    + self.records[("deformed", cycle, check.name)]
                )

        for i in range(72):
            self._append_detector(
                self.records[("deformed", self.CYCLES - 1, f"HX{i}")]
                + original_records[f"HX{i}"]
            )
            attached_edges = [e.index for e in self.lpu.edges if e.z_check == i]
            self._append_detector(
                self.records[("deformed", self.CYCLES - 1, f"HZ{i}")]
                + original_records[f"HZ{i}"]
                + [edge_records[e] for e in attached_edges]
            )
        for cycle_index, edge_indices in enumerate(self.lpu.cycles):
            self._append_detector(
                self.records[("deformed", self.CYCLES - 1, f"U{cycle_index}")]
                + [edge_records[e] for e in edge_indices]
            )

        for i, row in enumerate(self.code.hx):
            final = self._append_mpp("X", np.flatnonzero(row), ("final", f"HX{i}"))
            self._append_detector(original_records[f"HX{i}"] + [final])
        for i, row in enumerate(self.code.hz):
            final = self._append_mpp("Z", np.flatnonzero(row), ("final", f"HZ{i}"))
            self._append_detector(
                original_records[f"HZ{i}"]
                + [final]
                + self._correction_flip_records(np.flatnonzero(row), edge_records)
            )

        logical_measurement_records: list[int] = []
        for vertex in self.lpu.vertices:
            logical_measurement_records += self.records[("deformed", self.CYCLES - 1, self.vertex_name[vertex])]
        self._append_observable(0, logical_measurement_records)

        x_basis, z_basis = measurement_basis(self.code, self.mode)
        observable = 1
        for i in range(1, 12):
            ref = self.ref_qubits[i - 1]
            x_record = self._append_mpp(
                "X", [*np.flatnonzero(x_basis[i]), ref], ("logical", i, "X")
            )
            self._append_observable(observable, [x_record])
            observable += 1
            z_record = self._append_mpp(
                "Z", [*np.flatnonzero(z_basis[i]), ref], ("logical", i, "Z")
            )
            self._append_observable(observable, [z_record])
            observable += 1
        if observable != 23:
            raise AssertionError(observable)

    # Assemble the complete physical measurement circuit and its metadata.
    def build(self) -> BuiltCircuit:
        self._initial_state()
        self._edge_initialization()
        last_deformed_time, intervals = self._populate_deformed_timeline()
        self._emit_timeline(0, last_deformed_time)
        edge_records = self._measure_edges()
        original_records, _ = self._append_noisy_original_cycle()
        self._apply_gauge_correction(edge_records)
        self._add_detectors_and_observables(edge_records, original_records)

        dem = self.circuit.detector_error_model(
            decompose_errors=False,
            flatten_loops=True,
            allow_gauge_detectors=False,
        )
        metadata = {
            "schedule_method": {"graph_sync12": "synchronous_coloring_12_step_candidate"}[self.schedule],
            "schedule": self.schedule,
            "schedule_file": getattr(self, "schedule_path", Path("in_memory_schedule")).name,
            "coloring_seed": getattr(self, "schedule_payload", {}).get("coloring_seed", getattr(self, "schedule_payload", {}).get("seed")),
            "author_timestamps_available": False,
            "paper_baseline_verified": False,
            "schedule_certificate": getattr(self, "schedule_certificate", None),
            "lpu_vertices": len(self.lpu.vertices),
            "lpu_edges": len(self.lpu.edges),
            "lpu_cycles": len(self.lpu.cycles),
            "deformed_cycles": self.CYCLES,
            "dem_errors": dem.num_errors,
            "paper_fault_count": 324_526 if self.mode == "X" else 398_717,
        }
        return BuiltCircuit(
            mode=self.mode,
            p=self.p,
            circuit=self.circuit,
            fault_count=self.fault_count,
            num_qubits=self.num_qubits,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
            schedule_period=self.period,
            schedule_span=last_deformed_time + 1,
            metadata=metadata,
        )


class SingleXReconstructor(CircuitReconstructor):
    # Load the twelve-step coloring and check its connectivity and timing.
    def _load_optimized_schedule(self):
        from .schedule_certificate import certify_schedule
        self.schedule_path = Path(__file__).with_name('schedules') / 'sync12_X.json'
        self.schedule_payload = json.loads(self.schedule_path.read_text(encoding='utf-8'))
        tasks = [GateTask(**item) for item in self.schedule_payload['tasks']]
        expected, _ = self._schedule_template()
        key = lambda task: (task.gate, task.a, task.b, task.check_name)
        if Counter(map(key, tasks)) != Counter(map(key, expected)):
            raise ValueError('Schedule gate set differs from the circuit connectivity')
        actual = {key(task): task.time for task in tasks}
        shifts = {actual[key(task)] - (task.time - 3) for task in expected
                  if self.check_by_name[task.check_name].family in ('bb_x', 'bb_z')
                  and task.b < self.code.n}
        if len(shifts) != 1 or min(shifts) < 0:
            raise ValueError('BB gates must retain their relative timing')
        self.period = self.schedule_payload['repeat_period']
        self.schedule_certificate = certify_schedule(tasks, self.checks, self.period, self.CYCLES)
        if not self.schedule_certificate['valid']:
            raise ValueError(self.schedule_certificate['errors'])
        if self.schedule_certificate['single_cycle_span'] != self.period:
            raise ValueError('Each cycle must fill one repetition period')
        intervals = {}
        for check in self.checks:
            for ancilla in check.ancillas:
                times = [task.time for task in tasks if ancilla in (task.a, task.b)]
                intervals[f'{check.name}:{ancilla}'] = (min(times) - 1, max(times) + 1)
        return tasks, intervals

    # Append the Return checks with ideal gates, resets, and measurements.
    def _append_noisy_original_cycle(self):
        start, fault_count = len(self.circuit), self.fault_count
        records = super()._append_noisy_original_cycle()
        self.circuit = self.circuit[:start] + self.circuit[start:].without_noise()
        self.fault_count = fault_count
        return records

    # Build ten measurement cycles with the report and preserved logical observables.
    def build(self):
        built = super().build()
        built.metadata.update(schedule_method='a5_graph_coloring',
                              schedule='graph_sync12', boundary='ideal',
                              score_generators=23, target_report_observables=[0],
                              native_observables=list(range(23)),
                              final_target_observables=[])
        return built


# Build a single-X or joint-XX physical measurement circuit.
def build_circuit(code: GrossCode, lpu: LpuGraph, mode: str, p: float,
                  schedule: str = "graph_sync12") -> BuiltCircuit:
    if mode == 'X':
        return SingleXReconstructor(code, lpu, mode, p, schedule=schedule).build()
    return CircuitReconstructor(code, lpu, mode, p, schedule=schedule).build()
