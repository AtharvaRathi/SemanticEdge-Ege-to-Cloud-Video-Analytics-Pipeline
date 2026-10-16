"""
app/utils/hardware.py
=====================
Hardware Auto-Detection Module.

Probes available ONNX Runtime Execution Providers and returns an ordered list
so ORT can implement a proper per-op fallback chain:

  Priority order
  --------------
  1. OpenVINOExecutionProvider  → Intel CPUs / Integrated GPUs (iGPU)
     Only added if the *openvino.dll runtime* is actually loadable — not just
     if the ORT bridge DLL exists.  Uses GPU_FP16 by default; falls back to
     CPU_FP32 inside OpenVINO if the iGPU is unreachable.
  2. NnapiExecutionProvider     → Android / Exynos / ARM-based SoCs
     Independent 'if' block (not 'elif') so devices that report BOTH OpenVINO
     AND NNAPI get a full three-level chain: OpenVINO → NNAPI → CPU.
  3. CPUExecutionProvider       → Universal fallback (always appended last)

Key invariant
-------------
  len(providers) == len(provider_options) at all times.
  Both lists are built with .append() only — never .insert() — so index N in
  providers always aligns with index N in provider_options.

Usage
-----
    from app.utils.hardware import get_execution_providers, get_inference_session

    # Option A – use in ort.InferenceSession directly
    providers, provider_options = get_execution_providers()
    session = ort.InferenceSession(model_path, providers=providers,
                                   provider_options=provider_options)

    # Option B – convenience wrapper that returns a ready-made session
    session = get_inference_session(model_path)
"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


def _is_openvino_runtime_available() -> bool:
    """
    Check whether the OpenVINO *runtime* is actually loadable on this machine,
    not just whether the ORT bridge DLL exists.

    Problem: ort.get_available_providers() lists OpenVINOExecutionProvider
    whenever the onnxruntime_providers_openvino.dll is present, but that DLL
    itself depends on openvino.dll which may be missing if the user never
    installed the full OpenVINO runtime.  ORT then emits a scary error at
    session-creation time and falls back to CPU anyway — but the error log
    confuses users.

    Fix: try to import openvino.runtime.Core before listing the provider.
    If this import fails for any reason (DLL missing, wrong PATH, etc.)
    we skip the provider entirely and ORT goes straight to CPU with zero noise.
    """
    try:
        import openvino as ov   # noqa: F401
        _ = ov.Core()                          # forces the DLL chain to load now
        return True
    except Exception:
        return False


def get_execution_providers() -> tuple[list[str], list[dict]]:
    """
    Probe available ONNX Runtime Execution Providers and return the best
    ordered list together with any provider-specific options.

    IMPORTANT — alignment contract
    ------------------------------
    ORT reads providers[i] and provider_options[i] as a matched pair.
    Both lists are built exclusively with .append() so index N in providers
    always corresponds to index N in provider_options.  Never use .insert()
    on either list after the other has already been appended to.

    Returns
    -------
    providers : list[str]
        Ordered provider names (highest-priority first).
        ORT falls through to the next provider per-op when one fails.
    provider_options : list[dict]
        Parallel list of option dicts — len(providers) == len(provider_options)
        guaranteed.  Empty dicts mean "use defaults for this provider".
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        log.error("onnxruntime is not installed.")
        return ["CPUExecutionProvider"], [{}]

    log.info("Available ORT providers: %s", available)

    # Build both lists in strict lockstep — every .append() on one list is
    # immediately followed by an .append() on the other.
    providers:        list[str]  = []
    provider_options: list[dict] = []

    # ── 1. OpenVINO (Intel CPU / iGPU) ─────────────────────────────────────
    # Only added when BOTH the ORT bridge DLL AND the openvino.dll runtime are
    # loadable.  The pre-flight check avoids the noisy DLL-not-found error that
    # ORT would otherwise emit at session-creation time.
    if "OpenVINOExecutionProvider" in available and _is_openvino_runtime_available():
        log.info(
            "OpenVINOExecutionProvider: runtime OK → adding to provider chain "
            "(device=%s).", os.getenv("OPENVINO_DEVICE", "GPU_FP16")
        )
        providers.append("OpenVINOExecutionProvider")
        provider_options.append({                          # index matches providers[-1]
            "device_type": os.getenv("OPENVINO_DEVICE", "GPU_FP16"),
            "enable_opencl_throttling": "false",
            "cache_dir": os.getenv("OPENVINO_CACHE_DIR", ""),
        })
    elif "OpenVINOExecutionProvider" in available:
        # Bridge DLL found but runtime missing — skip silently
        log.info(
            "OpenVINOExecutionProvider listed by ORT but openvino.dll runtime not "
            "loadable. Skipping. Install the full OpenVINO runtime to enable iGPU."
        )

    # ── 2. NNAPI (Android / Exynos / ARM SoCs) ─────────────────────────────
    # Intentionally an independent 'if' (not 'elif') so devices that report
    # BOTH OpenVINO and NNAPI build the full three-provider chain:
    #   [OpenVINOExecutionProvider, NnapiExecutionProvider, CPUExecutionProvider]
    # ORT then tries ops in that priority order, giving maximum acceleration
    # coverage on exotic hardware while still landing on CPU for any unsupported op.
    if "NnapiExecutionProvider" in available:
        log.info("NnapiExecutionProvider: adding to provider chain (FP16 enabled).")
        providers.append("NnapiExecutionProvider")
        provider_options.append({                          # index matches providers[-1]
            # Allow FP16 compute on NNAPI-capable hardware (Exynos NPU, Mali GPU).
            "NNAPI_FLAG_USE_FP16": "1",
        })

    # ── 3. CPU fallback (always last) ──────────────────────────────────────
    providers.append("CPUExecutionProvider")
    provider_options.append({})                            # index matches providers[-1]

    # Sanity-check: misaligned lists would cause silent wrong-provider use.
    assert len(providers) == len(provider_options), (
        f"BUG: providers ({len(providers)}) and provider_options "
        f"({len(provider_options)}) are misaligned!"
    )

    log.info("Selected provider chain: %s", providers)
    return providers, provider_options


def get_inference_session(
    model_path: str,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    enable_memory_arena: bool = True,
) -> "ort.InferenceSession":  # noqa: F821  (type-hint only, not importing at module level)
    """
    Convenience function: build and return a hardware-optimised ORT
    InferenceSession for *model_path*.

    Parameters
    ----------
    model_path          : path to a .onnx file
    intra_op_threads    : threads for op-level parallelism (0 = ORT auto)
    inter_op_threads    : threads for graph-level parallelism (0 = ORT auto)
    enable_memory_arena : reduce allocator overhead on repeated calls

    Returns
    -------
    ort.InferenceSession configured with the best available provider.
    """
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()

    if intra_op_threads > 0:
        sess_opts.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        sess_opts.inter_op_num_threads = inter_op_threads
    if enable_memory_arena:
        sess_opts.enable_cpu_mem_arena = True

    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers, provider_options = get_execution_providers()

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=providers,
        provider_options=provider_options if any(provider_options) else None,
    )

    active = session.get_providers()[0]
    log.info("ORT session created for '%s' using provider: %s", model_path, active)
    return session
