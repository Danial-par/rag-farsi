from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rag_assistant.core.config import get_settings
from rag_assistant.core.event_loop import compatible_event_loop_factory
from rag_assistant.evaluation import load_dataset, load_observations, report_json, score_evaluation
from rag_assistant.live_evaluation import (
    DEFAULT_DATASET,
    EvaluationExecutionError,
    run_live_evaluation,
)


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
    """Run the real, credential-gated end-to-end sample evaluation."""
    parser = argparse.ArgumentParser(
        description="Run real retrieval, graph generation and persistence against the sample index"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--artifacts-dir", type=Path)
    arguments = parser.parse_args()
    kwargs = {}
    if arguments.case_ids is not None:
        kwargs["case_ids"] = arguments.case_ids
    if arguments.artifacts_dir is not None:
        kwargs["artifacts_dir"] = arguments.artifacts_dir
    try:
        report = asyncio.run(
            run_live_evaluation(
                settings=get_settings(), dataset=load_dataset(arguments.dataset), **kwargs
            ),
            loop_factory=compatible_event_loop_factory,
        )
    except EvaluationExecutionError as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "case_id": exc.case_id,
                    "failure_category": exc.error_code,
                }
            )
        )
        raise SystemExit(2) from None
    print(report_json(report))


if __name__ == "__main__":
    run()
