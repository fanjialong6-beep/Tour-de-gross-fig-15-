from __future__ import annotations
import time
import numpy as np
from .fault_model import FaultModel, grouped_priors

PARAMETERS = dict(alpha=1.0, gamma0=0.125, pre_iter=80, num_sets=600,
                  set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)


# Score a candidate correction against syndrome and observable residuals.
def classify(h, a, correction, detectors, observables) -> dict:
    predicted_d = np.asarray(h @ correction).ravel() % 2
    predicted_o = np.asarray(a @ correction).ravel() % 2
    residual = int(np.count_nonzero(predicted_d != detectors))
    wrong = np.flatnonzero(predicted_o != observables).tolist()
    return {"failure": bool(residual or wrong), "nonconverged": bool(residual),
            "converged_logical_failure": bool(not residual and wrong),
            "observable_failure": bool(wrong), "residual_syndrome_weight": residual,
            "wrong_observables": wrong, "correction_indices": np.flatnonzero(correction).tolist()}


# Decode one syndrome with Relay-BP-5 and independently score the correction.
def decode_one(model: FaultModel, detectors, observables, prior_p: float, seed: int, *, priors=None) -> dict:
    import relay_bp
    started = time.perf_counter()
    decoder_priors = (grouped_priors(model.multiplicity, prior_p) if priors is None
                      else np.asarray(priors, dtype=np.float64))
    if decoder_priors.shape != (model.h.shape[1],):
        raise ValueError('Decoder priors must have one entry per H column')
    if not np.isfinite(decoder_priors).all() or np.any((decoder_priors <= 0) | (decoder_priors >= 1)):
        raise ValueError('Decoder priors must be finite and strictly between zero and one')
    decoder = relay_bp.RelayDecoderF64(model.h, decoder_priors,
                                      seed=int(seed), **PARAMETERS)
    decoded = decoder.decode_detailed(np.asarray(detectors, dtype=np.uint8))
    correction = np.asarray(decoded.decoding, dtype=np.uint8)
    result = classify(model.h, model.a, correction, detectors, observables)
    result.update(iterations=int(decoded.iterations), decoder_reported_success=bool(decoded.success),
                  decoder_seed=int(seed), elapsed_seconds=time.perf_counter() - started)
    if bool(decoded.success) == result["nonconverged"]:
        raise RuntimeError("Decoder success flag contradicts independently checked Hc=sigma")
    return result
