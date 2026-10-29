from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from .gross_code import GrossCode, independent_rows, rank
from .lpu import Edge, LpuGraph, MERGED, build_lpu
from .schedule_certificate import certify_schedule
from .stim_circuit import (BuiltCircuit, Check, CircuitReconstructor, GateTask,
                          _edge_color, _pauli_string)


LEFT_ROOT = ("concurrent_left_root",)
RIGHT_ROOT = ("concurrent_right_root",)
SCHEDULE_PATH = Path(__file__).with_name("concurrent_schedule.json")


# Split the full LPU into independent X1 and X7 measurement components.
def build_concurrent_lpu(code: GrossCode) -> LpuGraph:
    full = build_lpu(code, "XX")
    left_data = set(np.flatnonzero(code.logical_x[0]).tolist())
    right_data = set(np.flatnonzero(code.logical_x[1]).tolist())
    # Map the shared root to the root of its measurement component.
    def node(v, kind):
        return (LEFT_ROOT if kind == "left" else RIGHT_ROOT) if v == MERGED else v
    left = {node(v, "left") for v in full.left_vertices}
    right = {node(v, "right") for v in full.right_vertices}
    vertices = sorted(left | right, key=repr)
    support = {}
    for v in vertices:
        if v == LEFT_ROOT:
            support[v] = sorted(left_data & set(full.support[MERGED]))
        elif v == RIGHT_ROOT:
            support[v] = sorted(right_data & set(full.support[MERGED]))
        else:
            support[v] = list(full.support[v])
    selected = [e for e in full.edges if e.kind != "bridge"]
    mapping = {e.index: i for i, e in enumerate(selected)}
    edges = [Edge(i, node(e.u, e.kind), node(e.v, e.kind), e.z_check, e.kind)
             for i, e in enumerate(selected)]
    cycles = [[mapping[e] for e in cycle] for cycle in full.cycles
              if all(e in mapping for e in cycle)]
    graph = LpuGraph("X1_X7_concurrent", vertices, edges, cycles, support, left, right)
    assert (len(vertices), len(edges), len(cycles)) == (24, 36, 8)
    assert not (left & right)
    assert all(len(q) == 1 for q in support.values())
    return graph


# Find independent gauge-correction trees in the two LPU components.
def forest_paths(graph: LpuGraph) -> dict:
    adjacency = defaultdict(list)
    for e in graph.edges:
        adjacency[e.u].append((e.v, e.index))
        adjacency[e.v].append((e.u, e.index))
    paths = {}
    for root in (LEFT_ROOT, RIGHT_ROOT):
        paths[root] = []
        queue = deque([root])
        while queue:
            v = queue.popleft()
            for other, e in adjacency[v]:
                if other not in paths:
                    paths[other] = paths[v] + [e]
                    queue.append(other)
    assert len(paths) == len(graph.vertices)
    return paths


class ConcurrentReconstructor(CircuitReconstructor):
    # Initialize the code, measurement geometry, and circuit state.
    def __init__(self, code: GrossCode, p: float,
                 eigenvalue_bits: tuple[int, int] = (0, 0)):
        self.eigenvalue_bits = tuple(int(x) for x in eigenvalue_bits)
        if len(self.eigenvalue_bits) != 2 or any(x not in (0, 1) for x in self.eigenvalue_bits):
            raise ValueError("two eigenvalue bits are required")
        super().__init__(code, build_concurrent_lpu(code), "XX", p, "graph_sync12")
        self.mode = "X1_X7_concurrent"

    # Allocate reference, edge, and check qubits with their supports.
    def _allocate_qubits_and_checks(self):
        self.ref_qubits = list(range(self.code.n, self.code.n + 10))
        cursor = self.code.n + 10
        self.edge_qubits = list(range(cursor, cursor + len(self.lpu.edges)))
        cursor += len(self.edge_qubits)
        self.checks, self.check_by_name, self.vertex_name = [], {}, {}
        # Allocate an ancilla and register one parity check.
        def add(name, pauli, data, family):
            nonlocal cursor
            data = list(data)
            check = Check(name, pauli, data, [cursor], [data.copy()], family)
            cursor += 1
            self.checks.append(check)
            self.check_by_name[name] = check
        for i, row in enumerate(self.code.hx):
            add(f"HX{i}", "X", np.flatnonzero(row), "bb_x")
        for i, row in enumerate(self.code.hz):
            data = list(np.flatnonzero(row)) + [self.edge_qubits[e.index] for e in self.lpu.edges if e.z_check == i]
            add(f"HZ{i}", "Z", data, "bb_z")
        for i, v in enumerate(self.lpu.vertices):
            name = f"V{i}"
            self.vertex_name[v] = name
            add(name, "X", self.lpu.support[v] + [self.edge_qubits[e] for e in self.lpu.incident_edges(v)], "vertex")
        for i, cycle in enumerate(self.lpu.cycles):
            add(f"U{i}", "Z", [self.edge_qubits[e] for e in cycle], "cycle")
        self.num_qubits = cursor
        self.physical_qubits = set(range(self.code.n)) | set(self.edge_qubits)
        self.physical_qubits |= {a for c in self.checks for a in c.ancillas}

    # Prepare the measured logical eigenstate and reference Bell pairs.
    def _initial_state(self):
        n = self.code.n + len(self.ref_qubits)
        stabilizers = [_pauli_string(n, x=np.flatnonzero(r)) for r in independent_rows(self.code.hx)]
        stabilizers += [_pauli_string(n, z=np.flatnonzero(r)) for r in independent_rows(self.code.hz)]
        for i in range(2):
            st = _pauli_string(n, x=np.flatnonzero(self.code.logical_x[i]))
            if self.eigenvalue_bits[i]:
                st *= -1
            stabilizers.append(st)
        for i in range(2, 12):
            ref = self.ref_qubits[i - 2]
            stabilizers.append(_pauli_string(n, x=[*np.flatnonzero(self.code.logical_x[i]), ref]))
            stabilizers.append(_pauli_string(n, z=[*np.flatnonzero(self.code.logical_z[i]), ref]))
        assert len(stabilizers) == n
        self.circuit += stim.Tableau.from_stabilizers(stabilizers).to_circuit("graph_state")

    # Construct the gate catalog and check-ancilla lifetimes.
    def _schedule_template(self):
        tasks = []
        offsets = {"bb_x": [(0,0),(0,1),(3,-1),(0,0),(1,0),(-1,-3)],
                   "bb_z": [(0,0),(-1,0),(1,3),(0,0),(0,-1),(-3,1)]}
        times = {"bb_x": self.X_TERM_TIMES, "bb_z": self.Z_TERM_TIMES}
        for check in self.checks:
            if check.family in offsets:
                x,y = divmod(int(check.name[2:]), self.code.m)
                for k,(dx,dy) in enumerate(offsets[check.family]):
                    q = self.code.index(0 if k < 3 else 1, x+dx, y+dy)
                    tasks.append(GateTask(times[check.family][k] + 3,
                                          "CX" if check.pauli == "X" else "CZ",
                                          check.ancillas[0], q, check.name))
        for check in self.checks:
            if check.family == "vertex":
                for q in check.data:
                    if q < self.code.n:
                        tasks.append(GateTask(1, "CX", check.ancillas[0], q, check.name))
        vertices = [(c.name,q) for c in self.checks if c.family == "vertex"
                    for q in c.data if q in self.edge_qubits]
        vcolors = _edge_color(vertices)
        for t, color in enumerate(vcolors, 2):
            for name,q in color:
                tasks.append(GateTask(t, "CX", self.check_by_name[name].ancillas[0], q, name))
        cycles = [(c.name,q) for c in self.checks if c.family == "cycle" for q in c.data]
        ccolors = _edge_color(cycles)
        for t,color in enumerate(ccolors, 2 + len(vcolors)):
            for name,q in color:
                tasks.append(GateTask(t, "CZ", self.check_by_name[name].ancillas[0], q, name))
        extras = [(f"HZ{e.z_check}",self.edge_qubits[e.index]) for e in self.lpu.edges if e.z_check is not None]
        ecolors = _edge_color(extras)
        start = max(9, 2 + len(vcolors) + len(ccolors))
        for t,color in enumerate(ecolors, start):
            for name,q in color:
                tasks.append(GateTask(t, "CZ", self.check_by_name[name].ancillas[0], q, name))
        self.period = max(t.time for t in tasks) + 2
        intervals = {}
        for check in self.checks:
            ts = [t.time for t in tasks if t.check_name == check.name]
            intervals[f"{check.name}:{check.ancillas[0]}"] = (min(ts)-1,max(ts)+1)
        self._validate_template(tasks, intervals)
        return tasks, intervals

    # Load and validate the concurrent X1/X7 gate schedule.
    def _load_optimized_schedule(self):
        expected, expected_intervals = self._schedule_template()
        period = self.period
        payload = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        if payload["mode"] != self.mode or payload["period"] != period or payload.get("bb_shift") != 3:
            raise ValueError("incompatible concurrent frozen schedule")
        tasks = [GateTask(**t) for t in payload["tasks"]]
        if tasks != expected:
            raise ValueError("concurrent schedule/source mismatch")
        self.schedule_certificate = certify_schedule(tasks,self.checks,period,cycles=self.CYCLES)
        if not self.schedule_certificate["valid"]:
            raise ValueError(self.schedule_certificate["errors"])
        self.schedule_path = SCHEDULE_PATH
        return tasks, expected_intervals

    # Apply measurement-controlled Pauli corrections after edge readout.
    def _apply_gauge_correction(self, edge_records):
        for vertex,path in forest_paths(self.lpu).items():
            targets = []
            for e in path:
                targets += [stim.target_rec(edge_records[e]-self.measurement_count), self.lpu.support[vertex][0]]
            if targets:
                self.circuit.append("CX", targets)

    # Find edge outcomes that flip a later Z-type measurement.
    def _correction_flip_records(self, z_support, edge_records):
        support = set(z_support)
        parity = np.zeros(len(self.lpu.edges),dtype=np.uint8)
        for vertex,path in forest_paths(self.lpu).items():
            if self.lpu.support[vertex][0] in support:
                for e in path:
                    parity[e] ^= 1
        return [edge_records[e] for e in np.flatnonzero(parity)]

    # Build syndrome detectors and logical scoring observables.
    def _add_detectors_and_observables(self, edge_records, original_records):
        for check in self.checks:
            if check.family != "vertex":
                self._append_detector(self.records[("deformed",0,check.name)])
        for cycle in range(1,self.CYCLES):
            for check in self.checks:
                self._append_detector(self.records[("deformed",cycle-1,check.name)] + self.records[("deformed",cycle,check.name)])
        for i in range(72):
            self._append_detector(self.records[("deformed",self.CYCLES-1,f"HX{i}")] + original_records[f"HX{i}"])
            self._append_detector(self.records[("deformed",self.CYCLES-1,f"HZ{i}")] + original_records[f"HZ{i}"] +
                                  [edge_records[e.index] for e in self.lpu.edges if e.z_check == i])
        for i,cycle in enumerate(self.lpu.cycles):
            self._append_detector(self.records[("deformed",self.CYCLES-1,f"U{i}")] + [edge_records[e] for e in cycle])
        for i,row in enumerate(self.code.hx):
            final = self._append_mpp("X",np.flatnonzero(row),("final",f"HX{i}"))
            self._append_detector(original_records[f"HX{i}"] + [final])
        for i,row in enumerate(self.code.hz):
            final = self._append_mpp("Z",np.flatnonzero(row),("final",f"HZ{i}"))
            self._append_detector(original_records[f"HZ{i}"] + [final] + self._correction_flip_records(np.flatnonzero(row),edge_records))
        self.readout_records = []
        for i,vertices in enumerate((self.lpu.left_vertices,self.lpu.right_vertices)):
            records = [r for v in sorted(vertices,key=repr) for r in self.records[("deformed",self.CYCLES-1,self.vertex_name[v])]]
            self.readout_records.append(records)
            self._append_observable(i,records)
        obs = 2
        for i in range(2,12):
            ref = self.ref_qubits[i-2]
            for pauli,rows in (("X",self.code.logical_x),("Z",self.code.logical_z)):
                record = self._append_mpp(pauli,[*np.flatnonzero(rows[i]),ref],("logical",i,pauli))
                self._append_observable(obs,[record])
                obs += 1
        assert obs == 22

    # Verify concurrent-check ranks, commutation, and measured logical products.
    def structure_report(self):
        cols = list(range(self.code.n)) + self.edge_qubits
        lookup = {q:i for i,q in enumerate(cols)}
        x,z = [],[]
        for check in self.checks:
            r = np.zeros(len(cols),dtype=np.uint8)
            r[[lookup[q] for q in check.data]] = 1
            (x if check.pauli == "X" else z).append(r)
        x,z = np.array(x),np.array(z)
        assert not ((x @ z.T)&1).any()
        rx,rz = rank(x),rank(z)
        assert len(cols)-rx-rz == 10
        for i,vertices in enumerate((self.lpu.left_vertices,self.lpu.right_vertices)):
            prod = np.zeros(len(cols),dtype=np.uint8)
            for v in vertices:
                for q in self.check_by_name[self.vertex_name[v]].data:
                    prod[lookup[q]] ^= 1
            assert np.array_equal(prod[:self.code.n],self.code.logical_x[i])
            assert not prod[self.code.n:].any()
        assert rx+rz-rank(self.code.hx)-rank(self.code.hz) == len(self.edge_qubits)+2
        return {"deformed_data_qubits":len(cols),"rank_x":rx,"rank_z":rz,
                "remaining_logical_qubits":10,"native_generators":22,
                "vertex_products":["X1","X7"],"commutation":True,
                "bb_blocks":1,"components":2,"shared_root_split":True,
                "root_weights":[len(self.check_by_name[self.vertex_name[v]].data) for v in (LEFT_ROOT,RIGHT_ROOT)],
                "root_ancillas":2,"root_bell_cx":0,
                "added_ancillas_relative_to_full_lpu":0,
                "added_couplers_relative_to_full_lpu":0,
                "active_lpu_qubits":len(self.edge_qubits)+32,
                "active_lpu_edge_data":len(self.edge_qubits),
                "active_lpu_check_ancillas":32,
                "shared_spectator_basis":"caller-provided canonical logical basis, indices 2..11; production uses explicit paper basis",
                "paper_concurrent_curve_exists":False}

    # Assemble the complete physical measurement circuit and its metadata.
    def build(self):
        structure = self.structure_report()
        self._initial_state()
        self._edge_initialization()
        last,intervals = self._populate_deformed_timeline()
        self._emit_timeline(0,last)
        edges = self._measure_edges()
        original,_ = self._append_noisy_original_cycle()
        self._apply_gauge_correction(edges)
        self._add_detectors_and_observables(edges,original)
        dem = self.circuit.detector_error_model(decompose_errors=False,flatten_loops=True,allow_gauge_detectors=False)
        assert dem.num_observables == 22
        tasks,_ = self._schedule_template()
        return BuiltCircuit(self.mode,self.p,self.circuit,self.fault_count,self.num_qubits,
            dem.num_detectors,dem.num_observables,self.period,last+1,
            {"schedule":"concurrent_split_roots_bb_shift3_v2","schedule_file":SCHEDULE_PATH.name,
             "bb_shift":3,"schedule_change":"original BB-to-data gates +2; all LPU couplings unchanged; noise retained",
             "schedule_sha256":hashlib.sha256(SCHEDULE_PATH.read_bytes()).hexdigest() if SCHEDULE_PATH.exists() else None,
             "schedule_certificate":self.schedule_certificate,"structure":structure,
             "lpu_vertices":24,"lpu_edges":36,"lpu_cycles":8,"deformed_cycles":self.CYCLES,
             "two_qubit_gates_per_deformed_cycle":len(tasks),"total_instruction_ticks":last+12,
             "author_timestamps_available":False,"paper_baseline_verified":False,
             "experimental_status":"new concurrent construction; frozen before noisy sampling",
             "eigenvalue_bits":list(self.eigenvalue_bits)})


# Build the concurrent X1 and X7 physical measurement circuit.
def build_concurrent_circuit(code: GrossCode, p: float) -> BuiltCircuit:
    return ConcurrentReconstructor(code,p).build()

