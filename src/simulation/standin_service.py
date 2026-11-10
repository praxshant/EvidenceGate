"""Configuration and factory for the local stand-in model service."""

from __future__ import annotations

import json
from pathlib import Path

from ..providers.standin_client import ReplayStore, StandinConfig, StandinProvider


def load_config(path: Path) -> StandinConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    return StandinConfig(**values)


def create_standin(root: Path, config_path: Path | None = None) -> StandinProvider:
    config = load_config(config_path) if config_path else StandinConfig()
    return StandinProvider(
        ReplayStore(root / "data" / "replay" / "responses.jsonl"),
        config,
    )
