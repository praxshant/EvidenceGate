"""Configuration and factory for the local stand-in model service."""

from __future__ import annotations

import json
from pathlib import Path

from ..providers.standin_client import ReplayStore, StandinConfig, StandinProvider


def load_config(path: Path) -> StandinConfig:
    from dataclasses import fields
    values = json.loads(path.read_text(encoding="utf-8"))
    valid_keys = {f.name for f in fields(StandinConfig)}
    filtered = {k: v for k, v in values.items() if k in valid_keys}
    return StandinConfig(**filtered)


def create_standin(root: Path, config_path: Path | None = None) -> StandinProvider:
    config = load_config(config_path) if config_path else StandinConfig()
    return StandinProvider(
        ReplayStore(root / "data" / "replay" / "responses.jsonl"),
        config,
    )
