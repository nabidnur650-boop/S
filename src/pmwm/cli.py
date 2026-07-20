from __future__ import annotations

import argparse
import time

from .common import ensure_directories, load_config

PILOT_STAGES = ["data", "features", "model", "memory", "evaluate", "figures"]
Q1_STAGES = ["data", "features", "backbone", "memory", "baselines", "evaluate", "online"]
FRESH_STAGES = ["predict", "evaluate", "online", "figures", "manuscript", "notebooks", "verify"]


def _through(requested: str, stages: list[str]) -> list[str]:
    if requested == "all":
        return stages
    return stages[: stages.index(requested) + 1]


def _run_pilot(stage: str, force: bool) -> None:
    from .data import stream_era5
    from .evaluation import run_evaluation
    from .features import prepare_features
    from .memory import build_memory
    from .model import generate_backbone_predictions, train_model
    from .plots import generate_figures

    config = load_config()
    ensure_directories()
    selected = _through(stage, PILOT_STAGES)
    if "data" in selected:
        stream_era5(config, force=force)
    if "features" in selected:
        prepare_features(config, force=force)
    if "model" in selected:
        train_model(config, force=force)
        generate_backbone_predictions(config, force=force)
    if "memory" in selected:
        build_memory(config, force=force)
    if "evaluate" in selected:
        run_evaluation(config, force=force)
    if "figures" in selected:
        generate_figures(config, force=force)


def _run_q1(stage: str, force: bool) -> None:
    from .q1_baselines import run_registered_neural_baselines
    from .q1_common import load_q1_config, verify_protocol_lock
    from .q1_data import stream_q1_era5
    from .q1_evaluation import run_q1_evaluation
    from .q1_features import prepare_q1_features
    from .q1_memory import build_all_q1_memories
    from .q1_model import train_and_encode_all_seeds
    from .q1_online import run_all_causal_online

    config = load_q1_config()
    verify_protocol_lock()
    selected = _through(stage, Q1_STAGES)
    if "data" in selected:
        stream_q1_era5(config, force=force)
    if "features" in selected:
        prepare_q1_features(config, force=force)
    if "backbone" in selected:
        train_and_encode_all_seeds(config, force=force)
    if "memory" in selected:
        build_all_q1_memories(config, force=force)
    if "baselines" in selected:
        run_registered_neural_baselines(config, force=force)
    if "evaluate" in selected:
        run_q1_evaluation(config, force=force)
    if "online" in selected:
        run_all_causal_online(config, force=force)


def _run_fresh(stage: str, force: bool) -> None:
    from .q1_manuscript import generate_manuscript
    from .q1_notebooks import execute_q1_notebooks
    from .q1_reporting import generate_q1_figures
    from .q1_v3 import evaluate_v3, run_all_v3_causal_online, train_and_predict_v3
    from .q1_verify import verify_q1_release

    selected = _through(stage, FRESH_STAGES)
    if "predict" in selected:
        train_and_predict_v3(force=force)
    if "evaluate" in selected:
        evaluate_v3(force=force)
    if "online" in selected:
        run_all_v3_causal_online(force=force)
    if "figures" in selected:
        generate_q1_figures()
    if "manuscript" in selected:
        generate_manuscript()
    if "notebooks" in selected:
        execute_q1_notebooks()
    if "verify" in selected:
        print(verify_q1_release())


def main() -> None:
    parser = argparse.ArgumentParser(description="PMWM ERA5 command-line interface")
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    pilot = subparsers.add_parser("pilot", help="run the original pilot workflow")
    pilot.add_argument("--stage", choices=["all", *PILOT_STAGES], default="all")
    pilot.add_argument("--force", action="store_true")
    q1 = subparsers.add_parser("q1", help="run the locked publication workflow")
    q1.add_argument("--stage", choices=["all", *Q1_STAGES], default="all")
    q1.add_argument("--force", action="store_true")
    fresh = subparsers.add_parser("fresh", help="run the fresh-confirmation v3 release workflow")
    fresh.add_argument("--stage", choices=["all", *FRESH_STAGES], default="all")
    fresh.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    began = time.perf_counter()
    if arguments.workflow == "pilot":
        _run_pilot(arguments.stage, arguments.force)
    elif arguments.workflow == "q1":
        _run_q1(arguments.stage, arguments.force)
    else:
        _run_fresh(arguments.stage, arguments.force)
    print(f"{arguments.workflow} workflow completed in {time.perf_counter() - began:.1f}s")


if __name__ == "__main__":
    main()
