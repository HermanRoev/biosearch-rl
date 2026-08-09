#!/usr/bin/env python3
"""Run the preregistered valid-but-wrong wind-direction diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from functools import partial
from math import degrees
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/biosearch-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/biosearch-cache")

from stable_baselines3 import PPO

from biosearch.ablations import (
    ZEROED_ODOR_FEATURES,
    ZEROED_WIND_FEATURES,
    AblationEpisode,
    ObservationCondition,
    PairedAblationComparison,
    ablation_episode_from_run,
    aggregate_ablation_episodes,
    compare_paired_ablation_outcomes,
    holm_adjusted_p_values,
    make_observation_transform,
    wind_rotation_for_seed,
)
from biosearch.evaluation import ControllerName, SensorCondition
from biosearch.experiments import evaluate_policy_batch, experiment_config
from biosearch.visualization.plots import save_wind_validity_comparison

INPUT_CONDITIONS = (
    ObservationCondition.UNMASKED,
    ObservationCondition.WIND_BLIND,
    ObservationCondition.WIND_ROTATED,
    ObservationCondition.ODOR_BLIND,
    ObservationCondition.ODOR_WIND_BLIND,
    ObservationCondition.ODOR_BLIND_WIND_ROTATED,
)
PAIRED_CONTRASTS = (
    (ObservationCondition.UNMASKED, ObservationCondition.WIND_BLIND),
    (ObservationCondition.UNMASKED, ObservationCondition.WIND_ROTATED),
    (ObservationCondition.ODOR_BLIND, ObservationCondition.ODOR_WIND_BLIND),
    (
        ObservationCondition.ODOR_BLIND,
        ObservationCondition.ODOR_BLIND_WIND_ROTATED,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 5.3 valid wind-vector control.")
    parser.add_argument(
        "--normal-model",
        type=Path,
        default=Path("models/best/best_model.zip"),
    )
    parser.add_argument(
        "--robust-model",
        type=Path,
        default=Path("models/robust/best/best_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=6_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def _write_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive")
    for model_path in (args.normal_model, args.robust_model):
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

    data_dir = args.results_dir / "data"
    figure_dir = args.results_dir / "figures"
    protocol_path = data_dir / "phase5_wind_validity_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    rotations = [degrees(wind_rotation_for_seed(seed)) for seed in seeds]
    protocol = {
        "phase": "5.3 valid-but-wrong wind-direction control",
        "question": (
            "Did wind masking hurt because the zero vector is out of distribution, "
            "or because PPO behavior depends on wind direction?"
        ),
        "hypothesis": (
            "If learned navigation uses wind direction, a unit-length wind vector "
            "rotated by 90 degrees should reduce success even though its encoding is valid."
        ),
        "physical_condition": SensorCondition.NORMAL.value,
        "policy_input_conditions": [condition.value for condition in INPUT_CONDITIONS],
        "factorial_design": {
            "odor": ["available", "hidden"],
            "wind": ["correct", "zero_vector", "valid_plus_or_minus_90_degrees"],
        },
        "episodes_per_policy_input_condition": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "max_steps": args.max_steps,
        "paired_design": (
            "All six input conditions use the same pose, simulation seed, physical "
            "plume, and time limit for a given policy and seed."
        ),
        "odor_mask": [*ZEROED_ODOR_FEATURES, "time_since_detection set to 1.0"],
        "zero_wind_mask": [*ZEROED_WIND_FEATURES, "both components set to 0.0"],
        "valid_wrong_wind": {
            "operation": "rotate the observed sine/cosine pair; do not change norm",
            "assignment": "-90 degrees for even seeds; +90 degrees for odd seeds",
            "negative_rotation_episodes": rotations.count(-90.0),
            "positive_rotation_episodes": rotations.count(90.0),
            "fixed_within_episode": True,
        },
        "predictions_declared_before_evaluation": {
            "robust_rotated_wind_with_odor": (
                "robust success drops by at least 30 percentage points versus correct "
                "wind, with Holm-adjusted p<0.05"
            ),
            "robust_rotated_wind_without_odor": (
                "odor-hidden robust success drops by at least 20 percentage points "
                "versus correct wind, with Holm-adjusted p<0.05"
            ),
            "zero_wind_without_odor_replication": (
                "robust success with odor and wind both hidden remains <=10%"
            ),
        },
        "statistics": (
            "95% Wilson intervals; two-sided exact McNemar tests for zero and rotated "
            "wind versus the matching correct-wind odor state; Holm correction over "
            "four tests within each policy"
        ),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote fixed protocol to {protocol_path}", flush=True)

    base_config = experiment_config(args.max_steps)
    models = {
        ControllerName.PPO_NORMAL: PPO.load(args.normal_model, device="cpu"),
        ControllerName.PPO_ROBUST: PPO.load(args.robust_model, device="cpu"),
    }
    episode_rows: list[AblationEpisode] = []
    for controller, model in models.items():
        for input_condition in INPUT_CONDITIONS:
            print(
                f"Evaluating {controller.value}: {input_condition.value}",
                flush=True,
            )
            runs = evaluate_policy_batch(
                model,
                controller_name=controller,
                condition=SensorCondition.NORMAL,
                seeds=seeds,
                base_config=base_config,
                observation_transform_factory=partial(
                    make_observation_transform,
                    condition=input_condition,
                ),
            )
            episode_rows.extend(ablation_episode_from_run(run, input_condition) for run in runs)

    summaries = aggregate_ablation_episodes(episode_rows)
    comparisons: list[PairedAblationComparison] = []
    for reference, comparison in PAIRED_CONTRASTS:
        comparisons.extend(
            compare_paired_ablation_outcomes(
                episode_rows,
                reference_condition=reference,
                comparison_condition=comparison,
            )
        )
    adjustments = holm_adjusted_p_values(comparisons)
    episode_dicts = [row.as_dict() for row in episode_rows]
    summary_dicts = [row.as_dict() for row in summaries]
    comparison_dicts = [
        {
            **row.as_dict(),
            "holm_adjusted_p_value": adjustments[(row.policy, row.comparison_condition)],
        }
        for row in comparisons
    ]
    _write_dicts(data_dir / "phase5_wind_validity_episodes.csv", episode_dicts)
    _write_dicts(data_dir / "phase5_wind_validity_summary.csv", summary_dicts)
    _write_dicts(data_dir / "phase5_wind_validity_paired.csv", comparison_dicts)

    summary_lookup = {(row.policy, row.observation_condition): row for row in summaries}
    comparison_lookup = {(row.policy, row.comparison_condition): row for row in comparisons}
    robust_policy = ControllerName.PPO_ROBUST.value
    robust_rotated = comparison_lookup[(robust_policy, ObservationCondition.WIND_ROTATED.value)]
    robust_odor_blind_rotated = comparison_lookup[
        (robust_policy, ObservationCondition.ODOR_BLIND_WIND_ROTATED.value)
    ]
    robust_no_cues = summary_lookup[(robust_policy, ObservationCondition.ODOR_WIND_BLIND.value)]
    analysis = {
        "protocol": str(protocol_path),
        "manipulation_check": {
            "rotation_degrees": sorted(set(rotations)),
            "wind_vector_norm": "preserved analytically by planar rotation",
            "physical_simulation_changed": False,
        },
        "predictions_supported": {
            "robust_rotated_wind_with_odor": (
                robust_rotated.success_rate_change_percentage_points <= -30.0
                and adjustments[(robust_policy, ObservationCondition.WIND_ROTATED.value)] < 0.05
            ),
            "robust_rotated_wind_without_odor": (
                robust_odor_blind_rotated.success_rate_change_percentage_points <= -20.0
                and adjustments[
                    (
                        robust_policy,
                        ObservationCondition.ODOR_BLIND_WIND_ROTATED.value,
                    )
                ]
                < 0.05
            ),
            "zero_wind_without_odor_replication": robust_no_cues.success_rate <= 10.0,
        },
        "paired_comparisons": comparison_dicts,
    }
    analysis_path = data_dir / "phase5_wind_validity_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    figure_path = figure_dir / "phase5_wind_validity_success.png"
    save_wind_validity_comparison(summaries, figure_path)

    for row in summaries:
        print(
            f"{row.policy:11s} {row.observation_condition:29s} "
            f"success={row.success_rate:5.1f}% "
            f"steps={row.mean_success_steps:6.1f}",
            flush=True,
        )
    for row in comparisons:
        print(
            f"{row.policy:11s} {row.reference_condition:11s}"
            f"->{row.comparison_condition:29s} "
            f"change={row.success_rate_change_percentage_points:+5.1f} pp "
            f"raw_p={row.exact_mcnemar_p_value:.6g} "
            f"holm_p={adjustments[(row.policy, row.comparison_condition)]:.6g}",
            flush=True,
        )
    print(f"Saved wind-validity figure to {figure_path}")
    print(f"Saved analysis to {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
