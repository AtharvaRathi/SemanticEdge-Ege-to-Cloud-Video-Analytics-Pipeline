"""
inference/engine.py
===================
OpenVINO-Enhanced Inference Engine (Item 5 of the optimisation plan).

This module exposes a thin wrapper around ONNX Runtime that specifically
targets the Intel OpenVINO Execution Provider to offload computation from the
i5-7300U CPU to the integrated Intel HD Graphics 620 (iGPU) using FP16 math.

Key features
------------
* Detects whether OpenVINO EP is available at runtime; falls back gracefully
  to CPU if not (same logic as app/utils/hardware.py but with more explicit
  OpenVINO-specific option tuning).
* GPU_FP16 device type → half-precision ops on iGPU (≈2× throughput vs FP32
  on CPU for transformer workloads).
* Kernel cache via ``cache_dir`` so compiled shaders are reused across process
  restarts (important for cold-start latency on edge devices).
* Exposes a ``run_inference(feed_dict)`` method that normalises inputs/outputs
  so callers don't need to know about provider-specific quirks.

Usage
-----
    from inference.engine import OpenVINOEngine

    engine = OpenVINOEngine("models/siglip_int8/model_quantized.onnx")
    embeddings = engine.run_inference({"pixel_values": pixel_np})

Environment variables
---------------------
    OPENVINO_DEVICE     Override target device (default: GPU_FP16)
                        Other options: CPU_FP32, GPU, AUTO
    OPENVINO_CACHE_DIR  Path to cache compiled OpenVINO kernels.
                        Default: <model_dir>/openvino_cache
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class OpenVINOEngine:
    """
    ONNX Runtime session wrapper with OpenVINO acceleration.

    On devices without OpenVINO (or on machines where the ORT-OpenVINO
    bridge is not installed) the session automatically falls back to
    CPUExecutionProvider – no code change needed.

    Parameters
    ----------
    model_path   : Absolute or relative path to the .onnx model file.
    device_type  : OpenVINO device string.
                   "GPU_FP16"  → Intel iGPU, half-precision (default)
                   "CPU_FP32"  → OpenVINO on CPU, full-precision
                   "AUTO"      → OpenVINO auto-selects best device
    cache_dir    : Directory for caching compiled OpenVINO kernels.
                   Defaults to a sibling ``openvino_cache`` folder next to
                   the model file.
    """

    def __init__(
        self,
        model_path: str,
        device_type: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path)
        self.device_type = device_type or os.getenv("OPENVINO_DEVICE", "GPU_FP16")

        # Default cache dir: sibling folder next to the model file
        if cache_dir is None:
            cache_dir = os.getenv(
                "OPENVINO_CACHE_DIR",
                str(Path(model_path).parent / "openvino_cache"),
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        available = ort.get_available_providers()
        log.info("Available ORT providers: %s", available)

        # ── Session options ─────────────────────────────────────────────────
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_cpu_mem_arena = True

        # ── Provider selection ──────────────────────────────────────────────
        if "OpenVINOExecutionProvider" in available:
            log.info(
                "OpenVINOExecutionProvider available → targeting device '%s'.",
                self.device_type,
            )
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            provider_options = [
                {
                    # Target Intel iGPU in FP16 mode for transformer throughput.
                    "device_type": self.device_type,
                    # Cache compiled kernels to cut cold-start time on restarts.
                    "cache_dir": self.cache_dir,
                    # Disable OpenCL CPU throttling so iGPU gets full bandwidth.
                    "enable_opencl_throttling": "false",
                    # Prefer static shapes for transformer encoders (faster graph
                    # compilation when input shapes are fixed).
                    "enable_dynamic_shapes": "false",
                },
                {},  # CPUExecutionProvider – use all defaults
            ]
        else:
            log.warning(
                "OpenVINOExecutionProvider not available. "
                "Install onnxruntime-openvino and the OpenVINO runtime to enable. "
                "Falling back to CPUExecutionProvider."
            )
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )

        self._input_names = {inp.name for inp in self.session.get_inputs()}
        self._output_names = [out.name for out in self.session.get_outputs()]
        self._active_provider = self.session.get_providers()[0]

        log.info(
            "OpenVINOEngine loaded '%s' via [%s]. Inputs: %s | Outputs: %s",
            os.path.basename(self.model_path),
            self._active_provider,
            sorted(self._input_names),
            self._output_names,
        )

    # -----------------------------------------------------------------------
    def run_inference(self, feed_dict: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Execute a forward pass.

        Parameters
        ----------
        feed_dict : dict mapping input name → numpy array.
                    Arrays are automatically cast to the dtype expected by the
                    model (float32 for pixel_values, int64 for token ids).

        Returns
        -------
        list of numpy arrays, one per model output (in declaration order).
        """
        # Sanitise feed dict: only include inputs the model actually accepts
        sanitised: dict[str, np.ndarray] = {}
        for name, value in feed_dict.items():
            if name not in self._input_names:
                log.debug("Skipping unexpected input '%s'.", name)
                continue
            if name == "pixel_values":
                sanitised[name] = np.asarray(value, dtype=np.float32)
            elif name in ("input_ids", "attention_mask"):
                sanitised[name] = np.asarray(value, dtype=np.int64)
            else:
                sanitised[name] = np.asarray(value)

        # Add any required inputs that are missing (zero-pad them)
        for inp in self.session.get_inputs():
            if inp.name not in sanitised:
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                dtype = np.float32 if inp.type == "tensor(float)" else np.int64
                log.debug("Auto-padding missing input '%s' with zeros.", inp.name)
                sanitised[inp.name] = np.zeros(shape, dtype=dtype)

        outputs = self.session.run(None, sanitised)
        return outputs

    # -----------------------------------------------------------------------
    @property
    def active_provider(self) -> str:
        """Return the name of the execution provider currently in use."""
        return self._active_provider

    @property
    def input_names(self) -> set[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names
