# src/export.py
"""
ONNX export + dynamic quantization + CPU benchmark (optional).

Design choice:
- This file is SAFE to import even if ONNX/optimum are not installed.
- The heavy imports happen only inside `export_run()`.

Notebook-friendly:
- call `export_run(run=..., root=..., seq_len=..., bench_n=...)`

Outputs under `models/onnx/<run>/`:
- model.onnx (fp32)
- model.int8.onnx (dynamic quantized)
- bench.json (sizes + latency stats)
- export.log
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from transformers import AutoTokenizer

from src.paths import ProjectPaths, setup_logger, write_json


def _bench_many(session, inputs: Dict[str, np.ndarray], n: int) -> Dict[str, float]:
    times: List[float] = []
    for _ in range(int(n)):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times, dtype=np.float64)
    return {
        "avg_sec": float(arr.mean()),
        "p50_sec": float(np.percentile(arr, 50)),
        "p95_sec": float(np.percentile(arr, 95)),
    }


def _size_mb(p: Path) -> float:
    return float(p.stat().st_size) / (1024.0 * 1024.0)


def export_run(
    run: str = "bert_lora",
    root: str | Path = ".",
    bench_n: int = 200,
    seq_len: int = 128,
    example_text: str = "User cannot connect to VPN on Windows 11 after update",
) -> Dict[str, Any]:
    paths = ProjectPaths.from_root(root)
    paths.ensure()

    run_dir = paths.models / run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    onnx_root = paths.models / "onnx" / run
    onnx_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(onnx_root / "export.log", "export")
    logger.info("Export run: %s", run)

    # Lazy imports (so import src.export never crashes)
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: optimum[onnxruntime]. "
            "Install: pip install -U \"optimum[onnxruntime]\" onnxruntime"
        ) from e

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: onnxruntime. Install: pip install -U onnxruntime"
        ) from e

    logger.info("Exporting model to ONNX")
    ort_model = ORTModelForTokenClassification.from_pretrained(str(run_dir), export=True)
    ort_model.save_pretrained(str(onnx_root))

    model_fp32 = onnx_root / "model.onnx"
    if not model_fp32.exists():
        candidates = list(onnx_root.glob("*.onnx"))
        if len(candidates) == 1:
            model_fp32 = candidates[0]
        else:
            raise FileNotFoundError("Could not locate exported ONNX model file in models/onnx/<run>/")

    logger.info("Quantizing ONNX model (dynamic INT8)")
    model_int8 = onnx_root / "model.int8.onnx"
    quantize_dynamic(str(model_fp32), str(model_int8), weight_type=QuantType.QInt8)

    tokenizer = AutoTokenizer.from_pretrained(str(run_dir), use_fast=True)
    enc = tokenizer(
        example_text,
        truncation=True,
        padding="max_length",
        max_length=int(seq_len),
        return_tensors="np",
    )
    inputs = {k: v for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}

    sess_opts = ort.SessionOptions()
    sess_fp32 = ort.InferenceSession(str(model_fp32), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(model_int8), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    for _ in range(20):
        sess_fp32.run(None, inputs)
        sess_int8.run(None, inputs)

    logger.info("Benchmarking fp32")
    stats_fp32 = _bench_many(sess_fp32, inputs, int(bench_n))
    logger.info("Benchmarking int8")
    stats_int8 = _bench_many(sess_int8, inputs, int(bench_n))

    out = {
        "run": run,
        "bench_n": int(bench_n),
        "seq_len": int(seq_len),
        "onnx_fp32_path": str(model_fp32),
        "onnx_int8_path": str(model_int8),
        "size_mb_fp32": float(_size_mb(model_fp32)),
        "size_mb_int8": float(_size_mb(model_int8)),
        "latency_fp32": stats_fp32,
        "latency_int8": stats_int8,
    }

    write_json(onnx_root / "bench.json", out)
    logger.info("Saved bench.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="bert_lora")
    ap.add_argument("--bench_n", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=128)
    args = ap.parse_args()
    out = export_run(run=args.run, root=".", bench_n=int(args.bench_n), seq_len=int(args.seq_len))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
