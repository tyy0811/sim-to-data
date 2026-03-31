"""Export trained CNN to ONNX for deployable inference.

Usage:
    python -m simtodata.export.onnx_export \
        --checkpoint models/best_B5.pt \
        --output models/best_B5.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def _model_device(model: torch.nn.Module) -> torch.device:
    """Infer device from first model parameter."""
    return next(model.parameters()).device


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    trace_length: int = 1024,
    opset_version: int = 14,
) -> str:
    """Export PyTorch model to ONNX.

    Args:
        model: trained CNN (any device — moved to CPU for export).
        output_path: where to save .onnx file.
        trace_length: input A-scan length (default 1024).
        opset_version: ONNX opset version.

    Returns:
        output_path.
    """
    model = model.cpu().eval()
    dummy_input = torch.randn(1, 1, trace_length)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["trace"],
        output_names=["logits"],
        dynamic_axes={
            "trace": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    return output_path


def verify_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    trace_length: int = 1024,
    n_samples: int = 10,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX output matches PyTorch within tolerance.

    Args:
        model: PyTorch model (any device — moved to CPU for comparison).
        onnx_path: path to exported ONNX model.
        trace_length: input length.
        n_samples: number of random samples to verify.
        atol: absolute tolerance.

    Returns:
        True if all samples match within tolerance.
    """
    import onnxruntime as ort

    model = model.cpu().eval()
    session = ort.InferenceSession(onnx_path)

    for _ in range(n_samples):
        x = torch.randn(1, 1, trace_length)
        with torch.no_grad():
            pt_out = model(x).numpy()
        onnx_out = session.run(None, {"trace": x.numpy()})[0]
        if not np.allclose(pt_out, onnx_out, atol=atol):
            return False
    return True


if __name__ == "__main__":
    from simtodata.models.factory import model_from_config

    parser = argparse.ArgumentParser(description="Export CNN to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model-config", default="configs/model_cnn1d.yaml")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--trace-length", type=int, default=1024)
    args = parser.parse_args()

    model = model_from_config(args.model_config)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    export_to_onnx(model, args.output, trace_length=args.trace_length)
    print(f"Exported: {args.output}")

    if verify_onnx(model, args.output, trace_length=args.trace_length):
        print("Verification: PASSED")
    else:
        print("Verification: FAILED — outputs diverge")
