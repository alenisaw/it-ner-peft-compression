# src/paths.py
"""
Project paths and directory initialization (+ small shared utilities).

Provides:
- ProjectPaths: resolves and creates required directories
- setup_logger(): consistent file+console logging
- read_yaml/read_json/write_json: simple IO helpers
- ds_key(): safe dataset key for Windows paths (replaces '/' with '__')
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def ds_key(name: str) -> str:
    return str(name).replace("/", "__")


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def setup_logger(log_path: Path, name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}::{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def find_repo_root(start: str | Path = ".") -> Path:
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent

    cur = p
    while True:
        if (cur / "configs" / "base.yaml").exists():
            return cur
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    raise FileNotFoundError(
        f"Repo root not found from start={p}. Expected 'configs/base.yaml' or '.git'."
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    configs_dir: Path
    config_path: Path

    data: Path
    models: Path
    report: Path
    raw_data: Path
    processed_data: Path
    figures: Path
    latex: Path

    onnx: Path
    logs: Path

    @staticmethod
    def from_root(root: str | Path = ".") -> "ProjectPaths":
        root = find_repo_root(root)

        configs_dir = root / "configs"
        config_path = configs_dir / "base.yaml"

        data = root / "data"
        models = root / "models"

        report = root / "notebooks" / "report"
        raw_data = data / "raw"
        processed_data = data / "processed"
        figures = report / "figures"
        latex = report / "latex"

        onnx = root / "onnx"
        logs = root / "logs"

        return ProjectPaths(
            root=root,
            configs_dir=configs_dir,
            config_path=config_path,
            data=data,
            models=models,
            report=report,
            raw_data=raw_data,
            processed_data=processed_data,
            figures=figures,
            latex=latex,
            onnx=onnx,
            logs=logs,
        )

    def ensure(self) -> None:
        for path in (
            self.configs_dir,
            self.data,
            self.models,
            self.report,
            self.raw_data,
            self.processed_data,
            self.figures,
            self.latex,
            self.onnx,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
