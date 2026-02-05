# src/export.py
"""
GPU-focused ONNX export and benchmark runner for token classification models.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from src.paths import ProjectPaths, setup_logger, write_json


def _set_reproducibility(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _latency_stats(times: List[float]) -> Dict[str, float]:
    arr = np.array(times, dtype=np.float64)
    return {
        "avg_sec": float(arr.mean()),
        "p50_sec": float(np.percentile(arr, 50)),
        "p95_sec": float(np.percentile(arr, 95)),
    }


def _size_mb(p: Path) -> float:
    return float(p.stat().st_size) / (1024.0 * 1024.0)


def _infer_base_model_from_adapter(run_dir: Path) -> str:
    cfg_path = run_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise RuntimeError("adapter_config.json not found for PEFT adapter")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path") or cfg.get("base_model_name") or cfg.get("base_model")
    if isinstance(base, str) and base.strip():
        return base.strip()
    raise RuntimeError("base_model_name_or_path is missing in adapter_config.json")


def _infer_num_labels_from_adapter(run_dir: Path, logger) -> int:
    st_path = run_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise RuntimeError("adapter_model.safetensors not found for PEFT adapter")
    try:
        from safetensors.torch import safe_open
    except Exception as e:
        raise RuntimeError("safetensors is required to inspect adapter weights") from e
    candidate_keys = []
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            lk = k.lower()
            if not lk.endswith(".weight"):
                continue
            if "lora_" in lk:
                continue
            if "classifier" in lk or "score" in lk:
                candidate_keys.append(k)
        prefer = None
        for k in candidate_keys:
            lk = k.lower()
            if lk.endswith("classifier.modules_to_save.default.weight") or lk.endswith(
                "score.modules_to_save.default.weight"
            ):
                prefer = k
                break
        if prefer is None:
            for k in candidate_keys:
                lk = k.lower()
                if lk.endswith("classifier.weight") or lk.endswith("score.weight"):
                    prefer = k
                    break
        key = prefer or (candidate_keys[0] if candidate_keys else None)
        if key is None:
            raise RuntimeError("classifier weight not found in adapter_model.safetensors")
        t = f.get_tensor(key)
        if t.ndim != 2:
            raise RuntimeError(f"classifier weight has unexpected ndim={t.ndim}")
        num_labels = int(t.shape[0])
    logger.info("Inferred num_labels from adapter: %d (key=%s)", num_labels, key)
    return num_labels


def _load_tokenizer(run_dir: Path, base_model: Optional[str]):
    try:
        return AutoTokenizer.from_pretrained(str(run_dir), use_fast=True)
    except Exception:
        if base_model:
            return AutoTokenizer.from_pretrained(base_model, use_fast=True)
    raise RuntimeError("Tokenizer not found in run_dir and base_model is unavailable")


def _load_model_any(run_dir: Path, logger) -> Tuple[torch.nn.Module, str]:
    try:
        cfg = AutoConfig.from_pretrained(str(run_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(run_dir), config=cfg)
        logger.info("Loaded full model from run_dir")
        return model, ""
    except Exception as e:
        logger.info("Full-model load failed: %s", e)

    adapter_weights = run_dir / "adapter_model.safetensors"
    adapter_cfg = run_dir / "adapter_config.json"
    if not (adapter_weights.exists() and adapter_cfg.exists()):
        raise RuntimeError(
            f"Could not load model from {run_dir}. No full model and no PEFT adapter found."
        )

    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft is required to load PEFT adapters") from e

    base_name = _infer_base_model_from_adapter(run_dir)
    num_labels = _infer_num_labels_from_adapter(run_dir, logger)
    cfg = AutoConfig.from_pretrained(base_name)
    cfg.num_labels = int(num_labels)
    base_model = AutoModelForTokenClassification.from_pretrained(base_name, config=cfg)
    peft_model = PeftModel.from_pretrained(base_model, str(run_dir))
    logger.info("Merging LoRA adapter into base model")
    merged = peft_model.merge_and_unload()
    return merged, base_name


def _accepts_token_type_ids(model: torch.nn.Module) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return True
    if "token_type_ids" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


class _Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, accept_token_type_ids: bool):
        super().__init__()
        self.model = model
        self.accept_token_type_ids = bool(accept_token_type_ids)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if self.accept_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.model(**kwargs)
        return out.logits


def _make_dummy_inputs(
    tokenizer,
    seq_len: int,
    example_text: str,
    device: torch.device,
    accept_token_type_ids: bool,
):
    enc = tokenizer(
        example_text,
        truncation=True,
        padding="max_length",
        max_length=int(seq_len),
        return_tensors="pt",
    )
    inputs = {}
    for k in ["input_ids", "attention_mask", "token_type_ids"]:
        if k in enc:
            if k == "token_type_ids" and not accept_token_type_ids:
                continue
            inputs[k] = enc[k].to(device)
    return inputs


def _supports_dynamic_shapes() -> bool:
    try:
        sig = inspect.signature(torch.onnx.export)
    except (TypeError, ValueError):
        return False
    return "dynamic_shapes" in sig.parameters


def _build_dynamic_shapes(input_names: List[str]) -> Dict[str, Dict[int, Any]]:
    from torch.export import Dim

    return {name: {0: Dim("batch"), 1: Dim("seq")} for name in input_names}


def _export_onnx(
    model: torch.nn.Module,
    tokenizer,
    out_path: Path,
    seq_len: int,
    example_text: str,
    opset: int,
    device: torch.device,
    dtype: torch.dtype,
    logger,
) -> Tuple[List[str], List[str]]:
    model = model.to(device)
    model.eval()
    model = model.to(dtype)
    accept_token_type_ids = _accepts_token_type_ids(model)
    wrapper = _Wrapper(model, accept_token_type_ids=accept_token_type_ids)
    dummy = _make_dummy_inputs(
        tokenizer,
        seq_len=seq_len,
        example_text=example_text,
        device=device,
        accept_token_type_ids=accept_token_type_ids,
    )
    input_names = list(dummy.keys())
    output_names = ["logits"]
    logger.info("torch.onnx.export -> %s", out_path)
    export_kwargs: Dict[str, Any] = {}
    if _supports_dynamic_shapes():
        export_kwargs["dynamic_shapes"] = _build_dynamic_shapes(input_names)
    else:
        dyn = {name: {0: "batch", 1: "seq"} for name in input_names}
        dyn["logits"] = {0: "batch", 1: "seq"}
        export_kwargs["dynamic_axes"] = dyn
    torch.onnx.export(
        wrapper,
        tuple(dummy.get(k) for k in input_names),
        f=str(out_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=int(opset),
        do_constant_folding=True,
        **export_kwargs,
    )
    return input_names, output_names


def _bench_torch(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    n: int,
    warmup: int,
    use_fp16: bool,
) -> Dict[str, float]:
    model.eval()
    times: List[float] = []
    with torch.no_grad():
        for _ in range(int(warmup)):
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = model(**inputs)
            else:
                _ = model(**inputs)
        torch.cuda.synchronize()
        for _ in range(int(n)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = model(**inputs)
            else:
                _ = model(**inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return _latency_stats(times)


def _bench_ort(session, inputs: Dict[str, np.ndarray], n: int, warmup: int) -> Dict[str, float]:
    for _ in range(int(warmup)):
        session.run(None, inputs)
    times: List[float] = []
    for _ in range(int(n)):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return _latency_stats(times)


def _profile_skipped(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def _profile_fallback(
    profile: Dict[str, Any],
    reason: str,
    provider: str,
    fallback_from: str,
) -> Dict[str, Any]:
    if not profile or profile.get("status") != "ok":
        return _profile_skipped(reason)
    out = dict(profile)
    out["fallback"] = True
    out["fallback_reason"] = reason
    out["fallback_from"] = fallback_from
    out["provider"] = provider
    return out


def export_run(
    run: str = "bert_lora",
    root: str | Path = ".",
    bench_n: int = 200,
    seq_len: int = 128,
    example_text: str = "User cannot connect to VPN on Windows 11 after update",
    opset: int = 18,
    export_fp32: bool = False,
) -> Dict[str, Any]:
    _set_reproducibility()
    paths = ProjectPaths.from_root(root)
    paths.ensure()

    run_dir = paths.models / run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    onnx_root = paths.models / "onnx" / run
    onnx_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(onnx_root / "export.log", "export")
    logger.info("Export run: %s", run)

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is required. Install: pip install -U onnxruntime") from e

    available_providers = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in available_providers
    trt_available = "TensorrtExecutionProvider" in available_providers
    need_cpu_ort = not cuda_available
    need_fp32_onnx = export_fp32 or need_cpu_ort

    model, base_name = _load_model_any(run_dir, logger)
    tokenizer = _load_tokenizer(run_dir, base_name)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fp16 ONNX export and GPU benchmarks")

    if int(opset) < 18:
        logger.warning("Requested opset %d < 18; using opset 18 to avoid exporter downgrades.", opset)
        opset = 18

    model_fp16_path = onnx_root / "model.fp16.onnx"
    input_names, output_names = _export_onnx(
        model=model,
        tokenizer=tokenizer,
        out_path=model_fp16_path,
        seq_len=int(seq_len),
        example_text=example_text,
        opset=int(opset),
        device=torch.device("cuda"),
        dtype=torch.float16,
        logger=logger,
    )

    model_fp32_path = None
    if need_fp32_onnx:
        model_fp32_path = onnx_root / "model.fp32.onnx"
        _export_onnx(
            model=model,
            tokenizer=tokenizer,
            out_path=model_fp32_path,
            seq_len=int(seq_len),
            example_text=example_text,
            opset=int(opset),
            device=torch.device("cpu"),
            dtype=torch.float32,
            logger=logger,
        )

    if not model_fp16_path.exists():
        raise FileNotFoundError("ONNX export failed: model.fp16.onnx not found")

    torch_device = torch.device("cuda")
    accept_token_type_ids = _accepts_token_type_ids(model)
    inputs_torch = _make_dummy_inputs(
        tokenizer,
        seq_len=seq_len,
        example_text=example_text,
        device=torch_device,
        accept_token_type_ids=accept_token_type_ids,
    )

    model = model.to(torch_device)
    model = model.float()
    torch_fp32 = {
        "status": "ok",
        "latency_sec": _bench_torch(model, inputs_torch, int(bench_n), warmup=30, use_fp16=False),
    }

    model = model.half()
    torch_fp16 = {
        "status": "ok",
        "latency_sec": _bench_torch(model, inputs_torch, int(bench_n), warmup=30, use_fp16=True),
    }

    enc = tokenizer(
        example_text,
        truncation=True,
        padding="max_length",
        max_length=int(seq_len),
        return_tensors="np",
    )
    ort_inputs = {k: enc[k] for k in input_names if k in enc}

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_cpu = _profile_skipped("CPUExecutionProvider not available or fp32 model missing")
    if "CPUExecutionProvider" in available_providers and model_fp32_path is not None:
        sess_cpu = ort.InferenceSession(
            str(model_fp32_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        ort_cpu = {
            "status": "ok",
            "latency_sec": _bench_ort(sess_cpu, ort_inputs, int(bench_n), warmup=30),
            "provider": "CPUExecutionProvider",
            "precision": "fp32",
        }

    ort_cuda = _profile_skipped("CUDAExecutionProvider not available")
    if cuda_available:
        sess_cuda = ort.InferenceSession(
            str(model_fp16_path),
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider"],
        )
        ort_cuda = {
            "status": "ok",
            "latency_sec": _bench_ort(sess_cuda, ort_inputs, int(bench_n), warmup=30),
            "provider": "CUDAExecutionProvider",
            "precision": "fp16",
        }
    elif ort_cpu.get("status") == "ok":
        ort_cuda = _profile_fallback(
            ort_cpu,
            "CUDAExecutionProvider not available; using CPUExecutionProvider",
            "CPUExecutionProvider",
            "ort_cpu_fp32",
        )

    ort_trt = _profile_skipped("TensorrtExecutionProvider not available")
    if trt_available:
        trt_cache = onnx_root / "trt_cache"
        trt_cache.mkdir(parents=True, exist_ok=True)
        trt_provider_options = {
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": str(trt_cache),
            "trt_fp16_enable": "1",
        }
        sess_trt = ort.InferenceSession(
            str(model_fp16_path),
            sess_options=sess_opts,
            providers=[
                ("TensorrtExecutionProvider", trt_provider_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        ort_trt = {
            "status": "ok",
            "latency_sec": _bench_ort(sess_trt, ort_inputs, int(bench_n), warmup=30),
            "provider": "TensorrtExecutionProvider",
            "precision": "fp16",
        }
    elif ort_cuda.get("status") == "ok":
        ort_trt = _profile_fallback(
            ort_cuda,
            "TensorrtExecutionProvider not available; using CUDAExecutionProvider",
            ort_cuda.get("provider", "CUDAExecutionProvider"),
            "ort_cuda_fp16",
        )
    elif ort_cpu.get("status") == "ok":
        ort_trt = _profile_fallback(
            ort_cpu,
            "TensorrtExecutionProvider not available; using CPUExecutionProvider",
            "CPUExecutionProvider",
            "ort_cpu_fp32",
        )

    out: Dict[str, Any] = {
        "run": run,
        "bench_n": int(bench_n),
        "seq_len": int(seq_len),
        "opset": int(opset),
        "input_names": input_names,
        "output_names": output_names,
        "onnx_fp16_path": str(model_fp16_path),
        "onnx_fp16_size_mb": float(_size_mb(model_fp16_path)),
        "onnx_fp32_path": str(model_fp32_path) if model_fp32_path else None,
        "onnx_fp32_size_mb": float(_size_mb(model_fp32_path)) if model_fp32_path else None,
        "providers": {
            "available": available_providers,
            "cuda_available": bool(cuda_available),
            "trt_available": bool(trt_available),
        },
        "device": {
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        },
        "profiles": {
            "torch_fp32_cuda": torch_fp32,
            "torch_fp16_cuda": torch_fp16,
            "ort_cpu_fp32": ort_cpu,
            "ort_cuda_fp16": ort_cuda,
            "ort_trt_fp16": ort_trt,
        },
    }

    write_json(onnx_root / "bench.json", out)
    logger.info("Saved bench.json -> %s", onnx_root / "bench.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="bert_lora")
    ap.add_argument("--bench_n", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--export_fp32", action="store_true")
    args = ap.parse_args()
    out = export_run(
        run=args.run,
        root=".",
        bench_n=int(args.bench_n),
        seq_len=int(args.seq_len),
        opset=int(args.opset),
        export_fp32=bool(args.export_fp32),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
