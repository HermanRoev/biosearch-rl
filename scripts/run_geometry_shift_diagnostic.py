#!/usr/bin/env python3
"""Run the preregistered source/start geometry-shift diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/biosearch-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/biosearch-cache")

from stable_baselines3 import PPO

from biosearch.ablations import (
    ZEROED_ODOR_FEATURES,
    ObservationCondition,
    apply_observation_condition,
)
from biosearch.evaluation import ControllerName, SensorCondition
from biosearch.experiments import evaluate_policy_batch, experiment_config
from biosearch.geometry import (
    GeometryCondition,
    GeometryEpisode,
    aggregate_geometry_episodes,
    compare_paired_geometry_outcomes,
    geometry_design_for_seed,
    geometry_episode_from_run,
    geometry_episode_setup,
    geometry_trajectory_from_run,
    holm_adjusted_geometry_p_values,
)
from biosearch.visualization.plots import (
    save_geometry_design_schematic,
    save_geometry_shift_comparison,
    save_geometry_trajectory_audit,
)

GEOMETRY_CONDITIONS = (
    GeometryCondition.SHIFTED_ALIGNED,
    GeometryCondition.CROSSWIND_DECOUPLED,
)
INPUT_CONDITIONS = (
    ObservationCondition.UNMASKED,
    ObservationCondition.ODOR_BLIND,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 5.4 geometry shift.")
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
    parser.add_argument("--seed-start", type=int, default=7_000)
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
    protocol_path = data_dir / "phase5_geometry_shift_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": "5.4 source/start geometry correlation test",
        "question": (
            "Do the frozen PPO policies still find a shifted source when the "
            "start's crosswind lane no longer predicts the source lane?"
        ),
        "hypothesis": (
            "If the wind-guided behavior relies on the fixed start/source geometry, "
            "success should collapse when source and start occupy opposite crosswind lanes."
        ),
        "models_frozen_before_design": True,
        "training_or_checkpoint_selection_performed": False,
        "physical_condition": SensorCondition.NORMAL.value,
        "geometry_conditions": [condition.value for condition in GEOMETRY_CONDITIONS],
        "policy_input_conditions": [condition.value for condition in INPUT_CONDITIONS],
        "episodes_per_cell": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "max_steps": args.max_steps,
        "source_assignment": ("x=2.5; y=3 for even seeds and y=9 for odd seeds (25 each at n=50)"),
        "aligned_start": ("x=16.5; source y plus deterministic Uniform[-1,1] crosswind jitter"),
        "decoupled_start": ("x=16.5; opposite y lane plus the identical per-seed crosswind jitter"),
        "heading": (
            "identical across paired geometry conditions; nominal upwind plus "
            "deterministic Uniform[-45,+45] degree jitter"
        ),
        "lane_balance": (
            "Both geometry conditions contain 25 starts in each lane marginally; "
            "only the source/start relationship changes."
        ),
        "obstacles": (
            "removed in both conditions to prevent obstacle contact from forcing or "
            "blocking the required crosswind correction"
        ),
        "odor_mask": [*ZEROED_ODOR_FEATURES, "time_since_detection set to 1.0"],
        "pairing": (
            "Within policy and input state, aligned and decoupled conditions share "
            "source, plume/sensor seed, heading, jitter draw, and time limit."
        ),
        "predictions_declared_before_evaluation": {
            "aligned_full_positive_control": (
                "both frozen policies achieve at least 60% with full input when "
                "source and start remain aligned in shifted lanes"
            ),
            "full_input_geometry_failure": (
                "each policy drops by at least 30 percentage points in decoupled "
                "geometry, with Holm-adjusted p<0.05"
            ),
            "robust_odor_blind_route_failure": (
                "robust odor-hidden aligned success is at least 25%, decoupled "
                "success is at most 10%, and the paired drop is at least 20 points "
                "with Holm-adjusted p<0.05"
            ),
        },
        "statistics": (
            "95% Wilson intervals; two-sided exact McNemar aligned-versus-decoupled "
            "tests; Holm correction over full and odor-hidden contrasts within policy"
        ),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote fixed protocol to {protocol_path}", flush=True)

    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    base_config = experiment_config(args.max_steps)
    models = {
        ControllerName.PPO_NORMAL: PPO.load(args.normal_model, device="cpu"),
        ControllerName.PPO_ROBUST: PPO.load(args.robust_model, device="cpu"),
    }
    episode_rows: list[GeometryEpisode] = []
    trajectory_rows = []
    for controller, model in models.items():
        for geometry_condition in GEOMETRY_CONDITIONS:
            for input_condition in INPUT_CONDITIONS:
                print(
                    f"Evaluating {controller.value}: {geometry_condition.value}, "
                    f"{input_condition.value}",
                    flush=True,
                )
                runs = evaluate_policy_batch(
                    model,
                    controller_name=controller,
                    condition=SensorCondition.NORMAL,
                    seeds=seeds,
                    base_config=base_config,
                    observation_transform=partial(
                        apply_observation_condition,
                        condition=input_condition,
                    ),
                    episode_setup_factory=partial(
                        geometry_episode_setup,
                        condition=geometry_condition,
                    ),
                )
                for run in runs:
                    design = geometry_design_for_seed(
                        run.metrics.seed,
                        base_config,
                        geometry_condition,
                    )
                    episode_rows.append(geometry_episode_from_run(run, design, input_condition))
                    if (
                        controller is ControllerName.PPO_ROBUST
                        and input_condition is ObservationCondition.ODOR_BLIND
                    ):
                        trajectory_rows.append(
                            geometry_trajectory_from_run(
                                run,
                                design,
                                input_condition,
                            )
                        )

    summaries = aggregate_geometry_episodes(episode_rows)
    comparisons = compare_paired_geometry_outcomes(episode_rows)
    adjustments = holm_adjusted_geometry_p_values(comparisons)
    episode_dicts = [row.as_dict() for row in episode_rows]
    summary_dicts = [row.as_dict() for row in summaries]
    comparison_dicts = [
        {
            **row.as_dict(),
            "holm_adjusted_p_value": adjustments[(row.policy, row.observation_condition)],
        }
        for row in comparisons
    ]
    _write_dicts(data_dir / "phase5_geometry_shift_episodes.csv", episode_dicts)
    _write_dicts(data_dir / "phase5_geometry_shift_summary.csv", summary_dicts)
    _write_dicts(data_dir / "phase5_geometry_shift_paired.csv", comparison_dicts)
    lane_groups: dict[tuple[str, str, str, str], list[GeometryEpisode]] = {}
    for row in episode_rows:
        source_lane = "lower" if row.source_y < 6.0 else "upper"
        key = (
            row.policy,
            row.geometry_condition,
            row.observation_condition,
            source_lane,
        )
        lane_groups.setdefault(key, []).append(row)
    lane_dicts = []
    for (policy, geometry, observation, source_lane), rows in sorted(lane_groups.items()):
        lane_dicts.append(
            {
                "policy": policy,
                "geometry_condition": geometry,
                "observation_condition": observation,
                "source_lane": source_lane,
                "episodes": len(rows),
                "success_rate": 100.0 * sum(row.success for row in rows) / len(rows),
                "mean_actual_odor_detection_percentage": sum(
                    row.actual_odor_detection_percentage for row in rows
                )
                / len(rows),
                "mean_final_distance_to_source": sum(row.final_distance_to_source for row in rows)
                / len(rows),
            }
        )
    _write_dicts(data_dir / "phase5_geometry_shift_lane_summary.csv", lane_dicts)

    summary_lookup = {
        (row.policy, row.geometry_condition, row.observation_condition): row for row in summaries
    }
    comparison_lookup = {(row.policy, row.observation_condition): row for row in comparisons}
    normal_policy = ControllerName.PPO_NORMAL.value
    robust_policy = ControllerName.PPO_ROBUST.value
    aligned = GeometryCondition.SHIFTED_ALIGNED.value
    decoupled = GeometryCondition.CROSSWIND_DECOUPLED.value
    full = ObservationCondition.UNMASKED.value
    odor_blind = ObservationCondition.ODOR_BLIND.value
    normal_full_comparison = comparison_lookup[(normal_policy, full)]
    robust_full_comparison = comparison_lookup[(robust_policy, full)]
    robust_blind_comparison = comparison_lookup[(robust_policy, odor_blind)]
    robust_blind_aligned = summary_lookup[(robust_policy, aligned, odor_blind)]
    robust_blind_decoupled = summary_lookup[(robust_policy, decoupled, odor_blind)]
    analysis = {
        "protocol": str(protocol_path),
        "design_precision_correction_after_evaluation": (
            "The opposite-lane condition reverses the source/start relationship; "
            "it breaks the trained positive alignment but is not statistically "
            "independent geometry. Continuous independently sampled coordinates "
            "remain a separate future test."
        ),
        "predictions_supported": {
            "aligned_full_positive_control": all(
                summary_lookup[(policy, aligned, full)].success_rate >= 60.0
                for policy in (normal_policy, robust_policy)
            ),
            "full_input_geometry_failure": all(
                comparison.success_rate_change_percentage_points <= -30.0
                and adjustments[(comparison.policy, full)] < 0.05
                for comparison in (normal_full_comparison, robust_full_comparison)
            ),
            "robust_odor_blind_route_failure": (
                robust_blind_aligned.success_rate >= 25.0
                and robust_blind_decoupled.success_rate <= 10.0
                and robust_blind_comparison.success_rate_change_percentage_points <= -20.0
                and adjustments[(robust_policy, odor_blind)] < 0.05
            ),
        },
        "prediction_components": {
            "normal_full_geometry_failure": (
                normal_full_comparison.success_rate_change_percentage_points <= -30.0
                and adjustments[(normal_policy, full)] < 0.05
            ),
            "robust_full_geometry_failure": (
                robust_full_comparison.success_rate_change_percentage_points <= -30.0
                and adjustments[(robust_policy, full)] < 0.05
            ),
        },
        "paired_comparisons": comparison_dicts,
        "post_hoc_mechanism_audit": {
            "status": "exploratory; not part of preregistered hypothesis tests",
            "robust_odor_blind": {
                geometry: {
                    "episodes": len(rows),
                    "crossed_world_centerline": sum(row.crossed_world_centerline for row in rows),
                    "mean_crosswind_span": sum(row.crosswind_span for row in rows) / len(rows),
                    "mean_closest_crosswind_distance_to_source_lane": sum(
                        row.closest_crosswind_distance_to_source_lane for row in rows
                    )
                    / len(rows),
                }
                for geometry in (aligned, decoupled)
                for rows in [
                    [
                        row
                        for row in episode_rows
                        if row.policy == robust_policy
                        and row.geometry_condition == geometry
                        and row.observation_condition == odor_blind
                    ]
                ]
            },
        },
    }
    analysis_path = data_dir / "phase5_geometry_shift_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    success_figure = figure_dir / "phase5_geometry_shift_success.png"
    design_figure = figure_dir / "phase5_geometry_design.png"
    trajectory_figure = figure_dir / "phase5_geometry_trajectory_audit.png"
    save_geometry_shift_comparison(summaries, success_figure)
    save_geometry_design_schematic(design_figure)
    save_geometry_trajectory_audit(trajectory_rows, trajectory_figure)

    for row in summaries:
        print(
            f"{row.policy:11s} {row.geometry_condition:21s} "
            f"{row.observation_condition:10s} success={row.success_rate:5.1f}% "
            f"steps={row.mean_success_steps:6.1f}",
            flush=True,
        )
    for row in comparisons:
        print(
            f"{row.policy:11s} {row.observation_condition:10s} "
            f"aligned->decoupled change="
            f"{row.success_rate_change_percentage_points:+5.1f} pp "
            f"raw_p={row.exact_mcnemar_p_value:.6g} "
            f"holm_p={adjustments[(row.policy, row.observation_condition)]:.6g}",
            flush=True,
        )
    print(f"Saved geometry result figure to {success_figure}")
    print(f"Saved geometry design figure to {design_figure}")
    print(f"Saved trajectory audit figure to {trajectory_figure}")
    print(f"Saved analysis to {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
