from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import sys

import numpy as np
from scipy.stats import beta

from .decoder import PARAMETERS, classify, decode_one
from .fault_model import FaultModel, grouped_priors
from .four_modes import MODES, metric_definition, shot_metrics

_WORKER_CONTEXT = None


# Derive an independent deterministic seed from the experiment identifiers.
def stable_seed(*parts):
    data = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int(hashlib.sha256(data).hexdigest()[:16], 16)


# Create the direct Monte Carlo cell identifier.
def cell_key(cell):
    return f"{cell['mode']}_p{cell['p']:.12g}_direct"


# Validate and normalize a fixed-sample experiment configuration.
def normalize_config(config):
    if not isinstance(config, dict) or not {"seed", "cells"} <= config.keys() or config.keys() - {"seed", "cells", "purpose"}:
        raise ValueError("Config requires seed and cells; purpose is optional")
    if type(config["seed"]) is not int or not isinstance(config["cells"], list) or not config["cells"]:
        raise ValueError("Require an integer seed and nonempty cells")
    if "purpose" in config and not isinstance(config["purpose"], str):
        raise ValueError("Purpose must be a string")
    cells = []
    for cell in config["cells"]:
        if not isinstance(cell, dict) or set(cell) != {"mode", "p", "shots"}:
            raise ValueError("Each cell requires mode, p, and shots")
        if cell["mode"] not in MODES or type(cell["shots"]) is not int or cell["shots"] <= 0:
            raise ValueError("Require a known mode and positive integer shots")
        if type(cell["p"]) not in (int, float) or not math.isfinite(cell["p"]) or not 0 < cell["p"] <= 1:
            raise ValueError("Physical p must be finite and in (0, 1]")
        cells.append({"mode": cell["mode"], "p": float(cell["p"]), "shots": cell["shots"]})
    if len({cell_key(c) for c in cells}) != len(cells):
        raise ValueError("Duplicate cell identifiers")
    return {"seed": config["seed"], "cells": cells, "purpose": config.get("purpose", "")}


# Publish a complete file with one atomic replacement.
def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


# Save a JSON document without nonfinite numeric values.
def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


# Allow only one running experiment to write an output directory.
@contextmanager
def output_lock(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(f".{output.name}.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This output directory is already in use") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# Load the requested models and compute their target-p decoder priors.
def load_models(config, model_root):
    models = {mode: FaultModel.load(Path(model_root) / mode)
              for mode in dict.fromkeys(c["mode"] for c in config["cells"])}
    for mode, model in models.items():
        if model.metadata.get("mode") != mode or model.a.shape[0] != len(metric_definition(mode)["score_observables"]):
            raise ValueError("Model mode or observable count does not match the experiment")
    priors = {cell_key(c): grouped_priors(models[c["mode"]].multiplicity, c["p"])
              for c in config["cells"]}
    return models, priors


# Identify the installed Relay native build using relative names and file hashes.
def relay_native_identity():
    distribution = importlib.metadata.distribution("relay-bp")
    binaries = {path.as_posix(): hashlib.sha256(Path(distribution.locate_file(path)).read_bytes()).hexdigest()
                for path in distribution.files or [] if path.suffix in (".pyd", ".so")}
    if not binaries:
        raise ValueError("Relay native binary could not be identified")
    return binaries


# Bind an experiment to its configuration, model contents, code, and dependencies.
def experiment_manifest(config, models, priors):
    root = Path(__file__).parent
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob("*")) if p.suffix in (".py", ".json")}
    value = {"schema": 1, "sampling": "independent_q_direct_mc", "config": config,
             "decoder_parameters": PARAMETERS,
             "models": {mode: model.content_hash() for mode, model in models.items()},
             "priors": {key: hashlib.sha256(p.astype("<f8").tobytes()).hexdigest() for key, p in priors.items()},
             "sources": sources,
             "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "stim", "relay-bp")},
             "relay_native_sha256": relay_native_identity(),
             "platform": {"system": platform.system(), "release": platform.release(),
                          "machine": platform.machine(), "python_implementation": platform.python_implementation()},
             "python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    return json.loads(json.dumps(value))


# Create or verify the immutable experiment manifest before resuming.
def prepare_output(output, manifest):
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Experiment configuration, code, dependencies, or model changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is nonempty and has no experiment manifest")
        write_json(path, manifest)


# Generate one independent fault sample with deterministic seeds.
def sample_shot(model, cell, master_seed, index):
    key = cell_key(cell)
    if cell["mode"] == "X1_single":
        streams = np.random.SeedSequence([master_seed, index]).spawn(3)
        rng = np.random.default_rng(streams[0])
        sample_seed = position_seed = int(streams[1].generate_state(1, dtype=np.uint64)[0])
        decoder_seed = int(streams[2].generate_state(1, dtype=np.uint64)[0])
    else:
        sample_seed = stable_seed(master_seed, key, index, "faults")
        position_seed = stable_seed(master_seed, key, index, "positions")
        decoder_seed = stable_seed(master_seed, key, index, "decoder")
        rng = np.random.default_rng(sample_seed)
    weight = int(rng.binomial(model.n, cell["p"] / 15))
    detectors, observables, ids = model.sample_weight(weight, position_seed)
    sample = {"cell": key, "index": index, "sample_seed": sample_seed,
              "position_seed": position_seed, "decoder_seed": decoder_seed,
              "raw_weight": weight, "raw_ids": ids.tolist(),
              "detector_indices": np.flatnonzero(detectors).tolist(),
              "observable_indices": np.flatnonzero(observables).tolist()}
    return sample, detectors, observables


# Locate the unique record for one cell and shot index.
def record_path(output, cell, index):
    return Path(output) / "records" / cell_key(cell) / f"shot_{index:08d}.json"


# Decode one sampled fault pattern and save its reproducible outcome.
def execute_shot(context, cell, index):
    output, config, models, priors = context
    model = models[cell["mode"]]
    sample, detectors, observables = sample_shot(model, cell, config["seed"], index)
    decoded = decode_one(model, detectors, observables, cell["p"], sample["decoder_seed"],
                         priors=priors[cell_key(cell)])
    record = {**sample, "model_sha256": model.metadata["model_sha256"], "decoder_prior_p": cell["p"],
              "decoded": decoded, "metrics": shot_metrics(decoded, cell["mode"])}
    write_json(record_path(output, cell, index), record)
    return index


# Load each process's models once and let the parent handle interruption.
def initialize_worker(output, config, model_root, expected_models):
    global _WORKER_CONTEXT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    models, priors = load_models(config, model_root)
    if {mode: model.content_hash() for mode, model in models.items()} != expected_models:
        raise ValueError("Model changed while starting workers")
    _WORKER_CONTEXT = (Path(output), config, models, priors)


# Dispatch one task using the process's persistent model objects.
def worker_shot(task):
    return execute_shot(_WORKER_CONTEXT, *task)


# Verify saved faults, corrections, and failure labels independently of Relay.
def verify_record(record, model, cell, seed, index):
    sample, detectors, observables = sample_shot(model, cell, seed, index)
    if any(record.get(key) != value for key, value in sample.items()):
        raise ValueError("Saved fault sample or seed changed")
    if record.get("model_sha256") != model.metadata["model_sha256"] or record.get("decoder_prior_p") != cell["p"]:
        raise ValueError("Saved model or decoder prior changed")
    decoded = record["decoded"]
    indices = decoded["correction_indices"]
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < model.h.shape[1] for i in indices) or indices != sorted(set(indices)):
        raise ValueError("Invalid saved correction")
    correction = np.zeros(model.h.shape[1], dtype=np.uint8)
    correction[indices] = 1
    checked = classify(model.h, model.a, correction, detectors, observables)
    if any(decoded.get(key) != value for key, value in checked.items()):
        raise ValueError("Saved failure classification changed")
    if decoded.get("decoder_seed") != sample["decoder_seed"] or decoded.get("decoder_reported_success") != (not checked["nonconverged"]):
        raise ValueError("Saved decoder identity or success flag changed")
    if record["metrics"] != shot_metrics(checked, cell["mode"]):
        raise ValueError("Saved metrics changed")
    return record["metrics"]


# Audit each saved record once and return missing indices and per-cell counts.
def collect_records(output, config, models):
    expected = {record_path(output, c, i): (c, i) for c in config["cells"] for i in range(c["shots"])}
    actual = set((output / "records").glob("*/*.json"))
    if actual - expected.keys():
        raise ValueError("Unexpected shot records in output directory")
    counts = {cell_key(c): {"completed": 0, "metrics": {key: 0 for key in
              shot_metrics({"nonconverged": False, "wrong_observables": []}, c["mode"])}}
              for c in config["cells"]}
    for path in sorted(actual):
        cell, index = expected[path]
        metrics = verify_record(json.loads(path.read_text(encoding="utf-8")), models[cell["mode"]],
                                cell, config["seed"], index)
        total = counts[cell_key(cell)]
        total["completed"] += 1
        for metric, failed in metrics.items():
            total["metrics"][metric] = total["metrics"].get(metric, 0) + int(failed)
    missing = [task for path, task in expected.items() if path not in actual]
    return missing, counts


# Calculate a binomial rate, standard error, and exact two-sided 95% interval.
def binomial_statistics(failures, shots):
    rate = failures / shots
    lower = 0.0 if failures == 0 else float(beta.ppf(.025, failures, shots - failures + 1))
    upper = 1.0 if failures == shots else float(beta.ppf(.975, failures + 1, shots - failures))
    return {"rate": rate, "standard_error": math.sqrt(rate * (1 - rate) / shots),
            "ci95_low": lower, "ci95_high": upper}


# Export audited counts and fixed-sample statistics as JSON and CSV.
def export_summary(output, config, counts, interrupted=False):
    cells, rows = [], []
    for cell in config["cells"]:
        count = counts[cell_key(cell)]
        complete = count["completed"] == cell["shots"]
        metrics = {key: {"failures": failures, **(binomial_statistics(failures, cell["shots"]) if complete else {})}
                   for key, failures in sorted(count["metrics"].items())}
        cells.append({**cell, "key": cell_key(cell), "completed": count["completed"], "complete": complete, "metrics": metrics})
        for metric, stats in metrics.items():
            rows.append({**cell, "completed": count["completed"], "complete": complete, "metric": metric, **stats})
    result = {"complete": all(c["complete"] for c in cells), "interrupted": interrupted,
              "primary_metric": "complete_instruction", "interval": "two-sided Clopper-Pearson 95%",
              "cells": cells}
    write_json(output / "summary.json", result)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["mode", "p", "shots", "completed", "complete", "metric", "failures", "rate", "standard_error", "ci95_low", "ci95_high"])
    writer.writeheader()
    writer.writerows(rows)
    write_text(output / "summary.csv", stream.getvalue())
    return result


# Execute bounded batches of process tasks while saving each completed shot.
def run_parallel(context, tasks, model_root, workers, manifest):
    output, config, _, _ = context
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(str(output), config, str(model_root), manifest["models"]))
    iterator, pending = iter(tasks), set()
    try:
        while True:
            while len(pending) < 2 * workers:
                task = next(iterator, None)
                if task is None:
                    break
                pending.add(executor.submit(worker_shot, task))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


# Run or resume a fixed-N experiment and audit all saved outcomes.
def run(config, output, model_root, workers=1):
    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer")
    config = normalize_config(config)
    output, model_root = Path(output).resolve(), Path(model_root).resolve()
    with output_lock(output):
        models, priors = load_models(config, model_root)
        manifest = experiment_manifest(config, models, priors)
        prepare_output(output, manifest)
        missing, counts = collect_records(output, config, models)
        context, interrupted = (output, config, models, priors), False
        export_summary(output, config, counts)
        try:
            if workers == 1:
                for cell, index in missing:
                    execute_shot(context, cell, index)
            elif missing:
                run_parallel(context, missing, model_root, workers, manifest)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            _, counts = collect_records(output, config, models)
            summary = export_summary(output, config, counts, interrupted)
    return summary
