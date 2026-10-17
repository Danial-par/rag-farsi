from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rag_assistant.core.config import get_settings
from rag_assistant.core.event_loop import compatible_event_loop_factory
from rag_assistant.evaluation import load_dataset, load_observations, report_json, score_evaluation
from rag_assistant.live_evaluation import DEFAULT_DATASET, run_live_evaluation


def run() -> None:
    """Score recorded RAG observations without calling external services."""
    parser = argparse.ArgumentParser(description="Score a recorded RAG evaluation run")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--k", type=int, default=5)
    arguments = parser.parse_args()
    report = score_evaluation(
        load_dataset(arguments.dataset), load_observations(arguments.observations), k=arguments.k
    )
    print(report_json(report))


def run_live() -> None:
    """Invoke OpenRouter for a manual, credential-gated sample evaluation."""
    parser = argparse.ArgumentParser(description="Run the sample evaluation using OpenRouter")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    arguments = parser.parse_args()
    report = asyncio.run(
        run_live_evaluation(settings=get_settings(), dataset=load_dataset(arguments.dataset)),
        loop_factory=compatible_event_loop_factory,
    )
    print(report_json(report))


if __name__ == "__main__":
    run()
