"""Batch inference with ONNX model.

Usage:
    python -m simtodata.export.onnx_infer \\
        --model models/best_B5.onnx \\
        --input data/sample_batch.npy
"""

from __future__ import annotations

import numpy as np


def run_inference(onnx_path: str, traces: np.ndarray) -> dict:
    """Run batch inference and return predictions with timing.

    Args:
        onnx_path: path to ONNX model.
        traces: (N, 1, L) input traces.

    Returns:
        Dict with predictions, probabilities, latency_ms.
    """
    import onnxruntime as ort
    import time

    session = ort.InferenceSession(onnx_path)

    start = time.perf_counter()
    logits = session.run(None, {"trace": traces.astype(np.float32)})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return {
        "predictions": probs.argmax(axis=1).tolist(),
        "probabilities": probs.tolist(),
        "n_samples": len(traces),
        "latency_ms": round(elapsed_ms, 2),
        "latency_per_sample_ms": round(elapsed_ms / len(traces), 4),
    }
