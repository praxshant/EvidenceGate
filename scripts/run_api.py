"""Run the FastAPI application with explicitly configured providers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from src.pipeline import ResearchPipeline
from src.providers.standin_client import ReplayStore, StandinProvider
from src.simulation.standin_service import create_standin


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    standin = create_standin(ROOT, ROOT / "data" / "standin" / "default.json")
    pipeline = ResearchPipeline(ROOT, {"stand-in": standin})
    from src.api import create_app

    uvicorn.run(create_app(pipeline), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
