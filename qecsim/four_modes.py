from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
import stim
from .fault_model import model_from_circuit
from .gross_code import measurement_basis
from .lpu import build_lpu
from .stim_circuit import build_circuit
from .xz_paper_basis import make_paper_gross_code, PAPER_LOGICAL_ORDER

MODES = ('X1_single', 'X1X7_joint', 'X1_X7_concurrent', 'X1Z7_joint')
P_GRID = [.002, .0025, .003, .0035, .004, .0045, .005]


# Add final target measurements to scoring observables only.
def complete_measurement_channel(built, code, mode):
    native_count = built.num_observables
    original_detectors = built.num_detectors
    targets = [stim.PauliString(code.n)]
    if mode == MODES[2]: targets.append(stim.PauliString(code.n))
    for q in np.flatnonzero(code.logical_x[0]): targets[0][int(q)] = 'X'
    if mode in (MODES[1], MODES[2]):
        for q in np.flatnonzero(code.logical_x[1]): targets[-1][int(q)] = 'X'
    if mode == MODES[3]:
        for q in np.flatnonzero(code.logical_z[1]): targets[0][int(q)] = 'Z'
    for i, target in enumerate(targets):
        terms = []
        for q in range(code.n):
            if target[q]:
                if terms: terms.append(stim.target_combiner())
                terms.append({1:stim.target_x, 2:stim.target_y, 3:stim.target_z}[target[q]](q))
        built.circuit.append('MPP', terms)
        built.circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], native_count + i)
    dem = built.circuit.detector_error_model(allow_gauge_detectors=False)
    assert dem.num_detectors == original_detectors
    built.num_detectors = dem.num_detectors
    built.num_observables = dem.num_observables
    built.metadata.update(final_target_consistency_detectors=0,
        final_target_observables=list(range(native_count, native_count + len(targets))),
        target_report_observables=list(range(len(targets))),
        native_observables=list(range(native_count)), score_generators=24,
        decoder_receives_final_target_measurements=False,
        primary_metric='complete_instruction',
        endpoint_convention='final targets in A only; score U OR R OR F OR G',
        boundary_author_equivalence='unverified')
    return built


# Build one of the four physical measurement channels.
def build_mode(mode: str, p: float = .003):
    code = make_paper_gross_code()
    if mode in MODES[:2]:
        old_mode = 'X' if mode == MODES[0] else 'XX'
        schedule = 'graph_sync12'
        built = build_circuit(code, build_lpu(code, old_mode), old_mode, p, schedule)
        built.mode = mode
        built.metadata['logical_basis'] = 'Appendix A.1'
    elif mode == MODES[2]:
        from .concurrent_mode import ConcurrentReconstructor
        built = ConcurrentReconstructor(code, p).build()
    elif mode == MODES[3]:
        from .mixed_xz_mode import build_xz_circuit
        built = build_xz_circuit(p, code=code)
    else:
        raise ValueError(mode)
    if mode != MODES[0]:
        built = complete_measurement_channel(built, code, mode)
    expected = 22 if mode == MODES[2] else 23
    assert built.num_observables == (23 if mode == MODES[0] else 24)
    built.metadata.update(mode=mode, paper_logical_order=list(PAPER_LOGICAL_ORDER),
        native_generators=expected, remaining_logical_qubits=10 if expected == 22 else 11,
        paper_baseline_verified=False,
        source_curve='Fig.15 direct X/XX' if mode in MODES[:2] else 'new experiment; no paper independent curve')
    return built


# Hash the scientific source files and frozen schedule resources.
def construction_signature():
    root = Path(__file__).resolve().parent
    names = ['four_modes.py', 'fault_model.py', 'gross_code.py', 'lpu.py', 'stim_circuit.py',
             'schedule_certificate.py', 'xz_paper_basis.py',
             'concurrent_mode.py', 'mixed_xz_mode.py', 'concurrent_schedule.json', 'xz_schedule.json',
             'schedules/sync12_X.json', 'schedules/sync12_XX.json']
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


# Compile a selected measurement channel into its grouped fault model.
def make_four_model(mode):
    built = build_mode(mode)
    noise = Counter()
    for inst in built.circuit.flattened():
        if inst.name in ('E', 'CORRELATED_ERROR'):
            noise['two_qubit_independent_pauli_events'] += 1
        elif inst.name in ('X_ERROR', 'Y_ERROR', 'Z_ERROR'):
            noise[inst.name + '_independent_events'] += len(inst.targets_copy())
    return model_from_circuit(built.circuit, built.fault_count, {
        **built.metadata, 'mode': mode, 'schedule_period': built.schedule_period,
        'schedule_span': built.schedule_span, 'qubits': built.num_qubits,
        'detectors': built.num_detectors, 'observables': built.num_observables,
        'noise_event_ledger': dict(noise),
        'physical_two_qubit_gates': noise['two_qubit_independent_pauli_events'] // 15,
        'metric_definition': metric_definition(mode),
        'construction_sources': construction_signature(),
    })


# Define report, final-target, and preserved-logical scoring rows.
def metric_definition(mode):
    if mode not in MODES:
        raise ValueError(mode)
    native_count = 22 if mode == MODES[2] else 23
    target_count = 2 if mode == MODES[2] else 1
    definition = {
        'primary_metric': 'complete_instruction',
        'native': 'nonconverged OR any legacy native observable flip (original 22 concurrent / 23 other rows; excludes added final-target rows)',
        'legacy_native': 'alias of native',
        'complete_instruction': 'nonconverged OR any of all 24 observable flips: report R, final target F, preserved generators G',
        'full_instruction': 'alias of complete_instruction',
        'logical_output_wrong_candidate': 'any R OR F OR G residual, independent of convergence; not full physical-state fidelity',
        'native_observables': list(range(native_count)),
        'target_report_observables': list(range(target_count)),
        'final_target_observables': list(range(native_count, 24)),
        'preserved_observables': list(range(target_count, native_count)),
        'score_observables': list(range(24)),
    }
    if mode == MODES[0]:
        definition.update(
            complete_instruction='nonconverged OR any of the 23 report and preserved observable flips',
            final_target_observables=[], score_observables=list(range(23)))
    if mode == MODES[1]:
        definition.update(parity_observables=[0], spectator_observables=list(range(3, 23)))
    elif mode == MODES[2]:
        definition.update(parity_observables=[0, 1], spectator_observables=list(range(2, 22)),
                          readout_observables=[0, 1])
    return definition


# Compute failure categories from one independently scored decoder outcome.
def shot_metrics(record, mode):
    nonconverged = bool(record['nonconverged'])
    wrong = set(record['wrong_observables'])
    definition = metric_definition(mode)
    if wrong - set(definition['score_observables']):
        raise ValueError('Observable index outside the selected score')
    native_wrong = bool(wrong.intersection(definition['native_observables']))
    report_wrong = bool(wrong.intersection(definition['target_report_observables']))
    final_wrong = bool(wrong.intersection(definition['final_target_observables']))
    preserved_wrong = bool(wrong.intersection(definition['preserved_observables']))
    complete = bool(nonconverged or wrong)
    result = {'native': bool(nonconverged or native_wrong),
        'legacy_native': bool(nonconverged or native_wrong),
        'complete_instruction': complete, 'full_instruction': complete,
        'nonconverged': nonconverged,
        'converged_logical': bool(not nonconverged and native_wrong),
        'converged_complete_instruction_failure': bool(not nonconverged and wrong),
        'logical_output_wrong_candidate': bool(wrong),
        'target_report_wrong_candidate': report_wrong,
        'final_target_wrong_candidate': final_wrong,
        'preserved_wrong_candidate': preserved_wrong,
        'converged_target_report_failure': bool(not nonconverged and report_wrong),
        'converged_final_target_failure': bool(not nonconverged and final_wrong),
        'converged_preserved_failure': bool(not nonconverged and preserved_wrong),
        'converged_non_target_logical_failure': bool(not nonconverged and preserved_wrong),
        'target_report_including_decode_failure': bool(nonconverged or report_wrong),
        'final_target_including_decode_failure': bool(nonconverged or final_wrong),
        'preserved_including_decode_failure': bool(nonconverged or preserved_wrong)}
    if 'parity_observables' in definition:
        parity = sum(i in wrong for i in definition['parity_observables']) % 2
        result['product_including_decode_failure'] = bool(nonconverged or parity)
        result['common21_including_decode_failure'] = bool(nonconverged or parity or
            any(i in wrong for i in definition['spectator_observables']))
    if mode == MODES[2]:
        a, b = 0 in wrong, 1 in wrong
        result.update(readout1_including_decode_failure=bool(nonconverged or a),
            readout7_including_decode_failure=bool(nonconverged or b),
            either_including_decode_failure=bool(nonconverged or a or b),
            converged_readout1=bool(not nonconverged and a),
            converged_readout7=bool(not nonconverged and b),
            converged_both_readouts=bool(not nonconverged and a and b))
    return result
