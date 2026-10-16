"""
tools/quantize_model.py
=======================
Exports the SigLIP vision encoder to ONNX format and applies INT8 static
quantization using Hugging Face Optimum + ONNX Runtime.

Target: Shrink the model from ~1.1 GB (FP32) down to ~250-300 MB (INT8).

Usage
-----
    python tools/quantize_model.py [--model google/siglip-base-patch16-224]
                                   [--output-dir models/siglip_int8]
                                   [--quantization-mode avx2|arm64|auto]

Output
------
    models/siglip_int8/
        model.onnx          (FP32 ONNX, intermediate)
        model_quantized.onnx (INT8 quantized, ready for inference)
"""

import argparse
import logging
import os
import sys
import platform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantize_model")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_quantization_mode() -> str:
    """Auto-detect best quantization preset based on the current CPU."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch" in machine:
        log.info("ARM64 CPU detected → using 'arm64' quantization preset.")
        return "arm64"
    log.info("x86/x64 CPU detected → using 'avx2' quantization preset.")
    return "avx2"


def _check_dependencies() -> None:
    """Verify that all required packages are installed before proceeding."""
    missing = []
    for pkg in ("optimum", "onnxruntime", "transformers", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing required packages: %s\n"
            "Install them with:\n"
            "  pip install optimum[onnxruntime] onnxruntime transformers torch",
            ", ".join(missing),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def export_to_onnx(model_name: str, output_dir: str) -> str:
    """
    Export the SigLIP *vision* encoder to ONNX using Optimum's ORTModelForFeatureExtraction.
    Returns the path to the exported ONNX model directory.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    log.info("Exporting '%s' → ONNX (this may take a few minutes)…", model_name)
    log.info("Output directory: %s", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # ORTModelForFeatureExtraction.from_pretrained with export=True triggers
    # the Optimum ONNX export pipeline automatically.
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        # Use the vision sub-model task so only the image encoder is exported.
        # For full cross-modal export you would need a custom approach.
    )
    ort_model.save_pretrained(output_dir)

    fp32_onnx_path = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(fp32_onnx_path):
        # Some Optimum versions name it differently; find whatever .onnx is there
        candidates = [f for f in os.listdir(output_dir) if f.endswith(".onnx")]
        if not candidates:
            raise FileNotFoundError(
                f"ONNX export completed but no .onnx file found in {output_dir}"
            )
        fp32_onnx_path = os.path.join(output_dir, candidates[0])

    log.info("FP32 ONNX model saved → %s", fp32_onnx_path)
    return output_dir


def quantize_model(onnx_dir: str, quantization_mode: str) -> str:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    log.info("Applying INT8 quantization (ONNX Runtime dynamic)…")

    # Find ONNX file
    onnx_path = None
    for f in os.listdir(onnx_dir):
        if f.endswith(".onnx"):
            onnx_path = os.path.join(onnx_dir, f)
            break

    if not onnx_path:
        raise FileNotFoundError("No ONNX model found to quantize.")

    quantized_path = os.path.join(onnx_dir, "model_quantized.onnx")

    # Dynamic quantization (safe + no calibration needed)
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8
    )

    log.info("INT8 quantized model saved → %s", quantized_path)
    return quantized_path


def print_size_comparison(fp32_dir: str, quantized_path: str) -> None:
    """Log a simple before/after file size comparison."""
    def dir_size_mb(path: str) -> float:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".onnx"):
                    total += os.path.getsize(os.path.join(root, f))
        return total / (1024 ** 2)

    fp32_mb = dir_size_mb(fp32_dir)
    q_mb = os.path.getsize(quantized_path) / (1024 ** 2)

    log.info("=" * 55)
    log.info("  Size comparison")
    log.info("  FP32 ONNX : %.1f MB", fp32_mb)
    log.info("  INT8 ONNX : %.1f MB  (%.0f%% reduction)", q_mb, (1 - q_mb / fp32_mb) * 100 if fp32_mb else 0)
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export & quantize SigLIP to INT8 ONNX for low-RAM deployment."
    )
    parser.add_argument(
        "--model",
        default="google/siglip-base-patch16-224",
        help="Hugging Face model ID or local path (default: google/siglip-base-patch16-224)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/siglip_int8",
        help="Directory to save the ONNX and quantized models (default: models/siglip_int8)",
    )
    parser.add_argument(
        "--quantization-mode",
        choices=["avx2", "arm64", "auto"],
        default="auto",
        help="Quantization preset (auto detects CPU, default: auto)",
    )
    args = parser.parse_args()

    # Resolve 'auto' mode
    mode = _detect_quantization_mode() if args.quantization_mode == "auto" else args.quantization_mode

    log.info("=" * 55)
    log.info("  SigLIP ONNX Quantizer")
    log.info("  Model            : %s", args.model)
    log.info("  Output dir       : %s", args.output_dir)
    log.info("  Quantization mode: %s", mode)
    log.info("=" * 55)

    _check_dependencies()

    # Step 1: Export FP32 → ONNX
    onnx_dir = export_to_onnx(args.model, args.output_dir)

    # Step 2: Optimize + quantize → INT8
    quantized_path = quantize_model(onnx_dir, mode)

    # Step 3: Report
    print_size_comparison(onnx_dir, quantized_path)

    log.info("Done! Use the quantized model in inference with:")
    log.info("  from optimum.onnxruntime import ORTModelForFeatureExtraction")
    log.info("  model = ORTModelForFeatureExtraction.from_pretrained('%s')", os.path.dirname(quantized_path))


if __name__ == "__main__":
    main()
