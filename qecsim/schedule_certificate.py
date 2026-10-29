from __future__ import annotations

from collections import defaultdict


# Validate qubit lifetimes and all within- and across-round check overlaps.
def certify_schedule(tasks, checks, repeat_period: int, cycles: int = 10,
                     paper_period: int = 12) -> dict:
    tasks, checks = list(tasks), list(checks)
    if not isinstance(repeat_period, int) or repeat_period < 1:
        raise ValueError('repeat_period must be a positive integer')
    if not isinstance(cycles, int) or cycles < 1:
        raise ValueError('cycles must be a positive integer')
    if not tasks or not checks:
        raise ValueError('tasks and checks must be nonempty')
    errors = []
    by_name = {c.name: c for c in checks}
    if len(by_name) != len(checks):
        errors.append('duplicate check names')
    ancillas = {int(a) for c in checks for a in c.ancillas}
    if len(ancillas) != sum(len(c.ancillas) for c in checks):
        errors.append('physical ancilla belongs to multiple checks')
    times = defaultdict(list)
    coupling = {}
    for task in tasks:
        a, b, time = int(task.a), int(task.b), int(task.time)
        if a == b:
            errors.append(f'gate at time {time} uses the same qubit twice')
        times[a].append(time)
        times[b].append(time)
        check = by_name.get(task.check_name)
        if check is None:
            errors.append(f'unknown check {task.check_name}')
            continue
        if a not in check.ancillas:
            errors.append(f'{check.name} gate control is not its physical ancilla')
        if b in check.data:
            key = (check.name, b)
            if key in coupling:
                errors.append(f'duplicate data coupling {key}')
            coupling[key] = time
        elif b not in check.ancillas:
            errors.append(f'{check.name} gate target is outside its support')

    ancilla_lifetimes, intervals, data_lifetimes = {}, {}, {}
    for check in checks:
        for q in check.data:
            if (check.name, int(q)) not in coupling:
                errors.append(f'missing data coupling {(check.name, int(q))}')
        for q in check.ancillas:
            q = int(q)
            if not times[q]:
                errors.append(f'ancilla {q} has no gates')
                continue
            start, end = min(times[q]) - 1, max(times[q]) + 1
            intervals[q] = (start, end)
            ancilla_lifetimes[str(q)] = {
                'check': check.name, 'reset_time': start, 'measurement_time': end,
                'first_gate': start + 1, 'last_gate': end - 1,
                'inclusive_lifetime': end - start + 1,
            }
        if len(check.ancillas) == 2:
            bell = [t for t in tasks if t.check_name == check.name
                    and t.a in check.ancillas and t.b in check.ancillas]
            interactions = [coupling[check.name, int(q)] for q in check.data
                            if (check.name, int(q)) in coupling]
            if len(bell) != 1 or not interactions or bell[0].time >= min(interactions):
                errors.append(f'invalid Bell preparation order for {check.name}')
    for q, qt in times.items():
        if q not in ancillas:
            data_lifetimes[str(q)] = max(qt) - min(qt) + 1
    if not intervals:
        raise ValueError('no scheduled physical ancillas')
    first = min(a for a, b in intervals.values())
    last = max(b for a, b in intervals.values())
    span = last - first + 1
    lifetime_bound = max([v['inclusive_lifetime'] for v in ancilla_lifetimes.values()]
                         + list(data_lifetimes.values()))

    occupied = {}
    collision_count = 0
    # Register a physical operation and detect simultaneous qubit use.
    def occupy(time, q, label):
        nonlocal collision_count
        key = (time, q)
        if key in occupied:
            collision_count += 1
            if collision_count <= 8:
                errors.append(f'qubit {q} at time {time}: {occupied[key]} conflicts with {label}')
        else:
            occupied[key] = label
    for cycle in range(cycles):
        offset = cycle * repeat_period
        for t in tasks:
            occupy(offset + int(t.time), int(t.a), f'gate {t.check_name} round {cycle}')
            occupy(offset + int(t.time), int(t.b), f'gate {t.check_name} round {cycle}')
        for q, (start, end) in intervals.items():
            occupy(offset + start, q, f'reset round {cycle}')
            occupy(offset + end, q, f'measurement round {cycle}')
            if cycles > 1 and end - start + 1 > repeat_period:
                message = f'ancilla {q} is reset before its previous round is measured'
                if message not in errors:
                    errors.append(message)

    overlap_pairs = 0
    cross_round_pairs = 0
    overlap_failures = 0
    for i, c in enumerate(checks):
        for other in checks[i + 1:]:
            if c.pauli == other.pauli:
                continue
            overlap = sorted(set(c.data) & set(other.data))
            if not overlap:
                continue
            if len(overlap) % 2:
                errors.append(f'anticommuting checks {c.name}, {other.name}')
                continue
            if any((c.name, int(q)) not in coupling or (other.name, int(q)) not in coupling for q in overlap):
                continue
            for r1 in range(cycles):
                for r2 in range(cycles):
                    signs = [coupling[c.name, int(q)] + r1 * repeat_period
                             - coupling[other.name, int(q)] - r2 * repeat_period for q in overlap]
                    overlap_pairs += 1
                    cross_round_pairs += r1 != r2
                    if any(v == 0 for v in signs) or sum(v < 0 for v in signs) % 2:
                        overlap_failures += 1
                        if overlap_failures <= 8:
                            errors.append(f'invalid overlap order {c.name}[{r1}], {other.name}[{r2}]: {signs}')

    reasons = []
    if errors:
        reasons.append('schedule fails physical or overlap validation')
    if span != paper_period:
        reasons.append(f'single-cycle span {span} differs from reported {paper_period}')
    if repeat_period != paper_period:
        reasons.append(f'repeat period {repeat_period} differs from reported {paper_period}')
    return {
        'repeat_period': repeat_period, 'cycles': cycles,
        'single_cycle_span': span, 'first_reset_time': first, 'last_measurement_time': last,
        'lifetime_period_bound': lifetime_bound,
        'total_span': (cycles - 1) * repeat_period + span,
        'lifetime_bound_below_span': lifetime_bound < span,
        'ancilla_lifetimes': ancilla_lifetimes, 'data_lifetimes': data_lifetimes,
        'event_collision_count': collision_count, 'overlap_pairs_checked': overlap_pairs,
        'cross_round_overlap_pairs_checked': cross_round_pairs,
        'overlap_failure_count': overlap_failures,
        'valid': not errors, 'errors': errors,
        'paper_timing_match': not reasons,
        'paper_graph_structure_match': None, 'mismatch_reasons': reasons,
        'author_exact_coloring_verified': False,
    }


# Certify a repeat period from the fixed gate and ancilla lifetimes.
def compact_schedule(tasks, checks, cycles: int = 10) -> dict:
    tasks, checks = list(tasks), list(checks)
    gate_times = [int(t.time) for t in tasks]
    roomy = max(gate_times) - min(gate_times) + 3
    initial = certify_schedule(tasks, checks, roomy, cycles)
    if not initial['valid']:
        raise ValueError('invalid input schedule: ' + '; '.join(initial['errors'][:3]))
    period = initial['lifetime_period_bound']
    certificate = certify_schedule(tasks, checks, period, cycles)
    if not certificate['valid']:
        raise ValueError('lifetime-bound repetition is invalid: ' + '; '.join(certificate['errors'][:3]))
    return {'period': period, 'certificate': certificate,
            'schedule_role': 'explicit_graph_compact_control'}
