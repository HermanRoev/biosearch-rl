"""Static Matplotlib figures for controller comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle as RectanglePatch

from biosearch.ablations import (
    AblationSummary,
    ObservationCondition,
    PairedAblationComparison,
)
from biosearch.config import SimulationConfig
from biosearch.evaluation import (
    MILESTONE4_CONDITIONS,
    AggregateMetrics,
    ControllerName,
    EpisodeRun,
    SensorCondition,
)
from biosearch.geometry import (
    GeometryCondition,
    GeometrySummary,
    GeometryTrajectory,
)

if TYPE_CHECKING:
    from biosearch.experiments import ExperimentResults

CONTROLLER_ORDER = (
    ControllerName.RANDOM.value,
    ControllerName.MOTH.value,
    ControllerName.PPO_NORMAL.value,
    ControllerName.PPO_ROBUST.value,
)
CONTROLLER_LABELS = {
    ControllerName.RANDOM.value: "Random",
    ControllerName.MOTH.value: "Moth-inspired",
    ControllerName.PPO_NORMAL.value: "Normal PPO",
    ControllerName.PPO_ROBUST.value: "Robust PPO",
}
CONTROLLER_COLORS = {
    ControllerName.RANDOM.value: "#94a3b8",
    ControllerName.MOTH.value: "#2563a8",
    ControllerName.PPO_NORMAL.value: "#f59e0b",
    ControllerName.PPO_ROBUST.value: "#16a34a",
}


def _draw_world(ax: Axes, config: SimulationConfig) -> None:
    world = config.world
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8fafc")
    for obstacle in world.obstacles:
        ax.add_patch(
            RectanglePatch(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                facecolor="#52606d",
                edgecolor="#263442",
                linewidth=1.2,
            )
        )
    ax.add_patch(
        Circle(
            world.source_position,
            world.source_radius,
            facecolor="#dc4331",
            edgecolor="#7f1d1d",
            linewidth=1.4,
            zorder=5,
        )
    )
    ax.annotate(
        "source",
        world.source_position,
        xytext=(0, -17),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="#7f1d1d",
    )
    wind = np.array([np.cos(world.wind_direction), np.sin(world.wind_direction)])
    ax.arrow(
        0.8,
        world.height - 0.8,
        1.3 * wind[0],
        1.3 * wind[1],
        width=0.035,
        head_width=0.25,
        head_length=0.35,
        color="#0e7490",
        length_includes_head=True,
    )
    ax.text(0.8, world.height - 1.25, "wind", color="#0e7490", fontsize=8)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.55)


def save_trajectory_comparison(
    runs: Sequence[EpisodeRun],
    config: SimulationConfig,
    output_path: Path,
    *,
    title: str = "Milestone 2: seeded example trajectories",
) -> None:
    """Save a grid of episode trajectories with odor-contact locations."""

    if not runs:
        raise ValueError("At least one episode run is required.")
    columns = 2
    rows = int(np.ceil(len(runs) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3.9 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, run in zip(axes.flat, runs, strict=False):
        _draw_world(ax, config)
        trajectory = run.trajectory
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#2563a8",
            linewidth=1.6,
            label="trajectory",
        )
        detected_points = trajectory[run.detections]
        if len(detected_points):
            ax.scatter(
                detected_points[:, 0],
                detected_points[:, 1],
                s=10,
                color="#f59e0b",
                edgecolors="none",
                alpha=0.75,
                label="odor detected",
                zorder=4,
            )
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            marker="^",
            s=55,
            color="#172554",
            label="start",
            zorder=6,
        )
        result_label = "success" if run.metrics.success else "timeout"
        controller_label = CONTROLLER_LABELS.get(
            run.metrics.controller,
            run.metrics.controller.title(),
        )
        ax.set_title(
            f"{controller_label}: "
            f"{run.metrics.condition.replace('_', ' ')}\n"
            f"{result_label}, {run.metrics.steps} steps, "
            f"{run.metrics.collisions} collisions"
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    for unused_ax in axes.flat[len(runs) :]:
        unused_ax.set_visible(False)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_metric_comparison(
    aggregates: Sequence[AggregateMetrics],
    output_path: Path,
) -> None:
    """Save success-rate and successful-search-time comparisons."""

    if not aggregates:
        raise ValueError("At least one aggregate metric is required.")
    controllers = sorted({row.controller for row in aggregates})
    conditions = sorted({row.condition for row in aggregates})
    lookup = {(row.controller, row.condition): row for row in aggregates}
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.36
    colors = {"moth": "#2563a8", "random": "#94a3b8"}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for index, controller in enumerate(controllers):
        offset = (index - (len(controllers) - 1) / 2) * width
        success_values = [lookup[(controller, condition)].success_rate for condition in conditions]
        step_values = [
            lookup[(controller, condition)].mean_success_steps for condition in conditions
        ]
        success_bars = axes[0].bar(
            x + offset,
            success_values,
            width,
            label=controller.title(),
            color=colors.get(controller),
        )
        axes[0].bar_label(
            success_bars,
            labels=[f"{value:.0f}%" for value in success_values],
            padding=3,
            fontsize=8,
        )
        plotted_step_values = [value if np.isfinite(value) else 0.0 for value in step_values]
        step_bars = axes[1].bar(
            x + offset,
            plotted_step_values,
            width,
            label=controller.title(),
            color=colors.get(controller),
        )
        axes[1].bar_label(
            step_bars,
            labels=[f"{value:.0f}" if np.isfinite(value) else "n/a" for value in step_values],
            padding=3,
            fontsize=8,
        )

    labels = [condition.replace("_", " ").title() for condition in conditions]
    axes[0].set_title("Success rate")
    axes[0].set_ylabel("Successful episodes (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_title("Mean search time")
    axes[1].set_ylabel("Steps (successful episodes only)")
    for condition_index, condition in enumerate(conditions):
        values = [lookup[(controller, condition)].mean_success_steps for controller in controllers]
        if not any(np.isfinite(value) for value in values):
            axes[1].text(
                condition_index,
                12,
                "no successful\nepisodes",
                ha="center",
                va="bottom",
                color="#64748b",
                fontsize=8,
            )
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    episode_counts = {row.episodes for row in aggregates}
    sample_note = (
        f" (n={next(iter(episode_counts))} per controller/condition)"
        if len(episode_counts) == 1
        else ""
    )
    figure.suptitle(
        f"Milestone 2: random vs. moth-inspired controller{sample_note}",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_training_curve(
    evaluations_path: Path,
    output_path: Path,
) -> None:
    """Plot periodic PPO evaluation reward and success rate."""

    if not evaluations_path.exists():
        raise FileNotFoundError(f"Training evaluations not found: {evaluations_path}")
    with np.load(evaluations_path) as data:
        timesteps = data["timesteps"]
        rewards = data["results"]
        successes = data["successes"]

    mean_rewards = rewards.mean(axis=1)
    reward_std = rewards.std(axis=1)
    success_rates = 100.0 * successes.mean(axis=1)
    figure, reward_axis = plt.subplots(
        figsize=(9.5, 4.8),
        constrained_layout=True,
    )
    success_axis = reward_axis.twinx()
    reward_axis.plot(
        timesteps,
        mean_rewards,
        color="#2563a8",
        marker="o",
        linewidth=2,
        label="Mean evaluation reward",
    )
    reward_axis.fill_between(
        timesteps,
        mean_rewards - reward_std,
        mean_rewards + reward_std,
        color="#2563a8",
        alpha=0.14,
        linewidth=0,
    )
    success_axis.plot(
        timesteps,
        success_rates,
        color="#dc4331",
        marker="s",
        linewidth=2,
        label="Evaluation success rate",
    )
    reward_axis.set_xlabel("Training timesteps")
    reward_axis.set_ylabel("Mean episode reward", color="#2563a8")
    success_axis.set_ylabel("Success rate (%)", color="#dc4331")
    success_axis.set_ylim(-3, 103)
    reward_axis.grid(color="#cbd5e1", linewidth=0.7, alpha=0.7)
    reward_axis.spines["top"].set_visible(False)
    success_axis.spines["top"].set_visible(False)
    reward_axis.tick_params(axis="y", colors="#2563a8")
    success_axis.tick_params(axis="y", colors="#dc4331")
    lines = [*reward_axis.lines, *success_axis.lines]
    reward_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower right",
        frameon=False,
    )
    figure.suptitle(
        f"PPO learning curve ({rewards.shape[1]} seeded episodes per evaluation)",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_milestone4_metric_plot(
    aggregates: Sequence[AggregateMetrics],
    output_path: Path,
    *,
    metric: str,
) -> None:
    """Plot one aggregate metric over all required evaluation conditions."""

    if metric not in {"success_rate", "mean_success_steps"}:
        raise ValueError("metric must be success_rate or mean_success_steps.")
    lookup = {(row.controller, row.condition): row for row in aggregates}
    conditions = [
        condition.value
        for condition in MILESTONE4_CONDITIONS
        if any(key[1] == condition.value for key in lookup)
    ]
    controllers = [
        controller for controller in CONTROLLER_ORDER if any(key[0] == controller for key in lookup)
    ]
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.19
    figure, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    for index, controller in enumerate(controllers):
        values = [
            float(getattr(lookup[(controller, condition)], metric)) for condition in conditions
        ]
        plotted_values = [value if np.isfinite(value) else 0.0 for value in values]
        offset = (index - (len(controllers) - 1) / 2) * width
        error_bars = None
        if metric == "success_rate":
            rows = [lookup[(controller, condition)] for condition in conditions]
            error_bars = np.asarray(
                [
                    [max(0.0, row.success_rate - row.success_ci95_low) for row in rows],
                    [max(0.0, row.success_ci95_high - row.success_rate) for row in rows],
                ],
                dtype=np.float64,
            )
        bars = ax.bar(
            x + offset,
            plotted_values,
            width,
            color=CONTROLLER_COLORS[controller],
            label=CONTROLLER_LABELS[controller],
            yerr=error_bars,
            capsize=2 if error_bars is not None else 0,
            error_kw={"linewidth": 0.8},
        )
        if metric == "success_rate":
            ax.bar_label(
                bars,
                labels=[f"{value:.0f}" for value in values],
                padding=2,
                fontsize=7,
                rotation=90,
            )
        else:
            for condition_index, value in enumerate(values):
                if not np.isfinite(value):
                    ax.text(
                        x[condition_index] + offset,
                        7,
                        "no success",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6.5,
                        color="#64748b",
                    )

    labels = [condition.replace("_", " ").title() for condition in conditions]
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "success_rate":
        ax.legend(frameon=False, ncols=len(controllers), loc="upper center")
        ax.set_ylim(0, 112)
        ax.set_ylabel("Successful episodes (%)")
        title = "Success rate across six evaluation conditions (95% Wilson CI)"
    else:
        ax.legend(frameon=False, ncols=2, loc="upper right")
        ax.set_ylim(0, 450)
        ax.set_ylabel("Steps among successful episodes")
        title = "Mean search time (successful episodes only)"
    sample_sizes = {row.episodes for row in aggregates}
    sample_note = (
        f" (n={next(iter(sample_sizes))} per controller/condition)"
        if len(sample_sizes) == 1
        else ""
    )
    figure.suptitle(
        f"{title}{sample_note}",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_sensor_failure_comparison(
    aggregates: Sequence[AggregateMetrics],
    output_path: Path,
) -> None:
    """Show the effect of complete left-sensor loss on success rate."""

    lookup = {(row.controller, row.condition): row for row in aggregates}
    controllers = [
        controller
        for controller in CONTROLLER_ORDER
        if (controller, SensorCondition.NORMAL.value) in lookup
        and (controller, SensorCondition.LEFT_DISABLED.value) in lookup
    ]
    x = np.arange(len(controllers), dtype=np.float64)
    width = 0.34
    figure, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    series = (
        (SensorCondition.NORMAL.value, "Normal sensors", "#38bdf8"),
        (SensorCondition.LEFT_DISABLED.value, "Left sensor disabled", "#dc4331"),
    )
    for index, (condition, label, color) in enumerate(series):
        rows = [lookup[(controller, condition)] for controller in controllers]
        values = [row.success_rate for row in rows]
        error_bars = np.asarray(
            [
                [max(0.0, row.success_rate - row.success_ci95_low) for row in rows],
                [max(0.0, row.success_ci95_high - row.success_rate) for row in rows],
            ],
            dtype=np.float64,
        )
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
            yerr=error_bars,
            capsize=3,
            error_kw={"linewidth": 0.9},
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.0f}%" for value in values],
            padding=3,
            fontsize=8,
        )
    ax.set_xticks(x, [CONTROLLER_LABELS[controller] for controller in controllers])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Successful episodes (%)")
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    figure.suptitle(
        "Robustness to complete loss of the left odor sensor (95% Wilson CI)",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_visitation_heatmap(
    results: ExperimentResults,
    config: SimulationConfig,
    output_path: Path,
    *,
    condition: SensorCondition = SensorCondition.LEFT_DISABLED,
) -> None:
    """Plot normalized trajectory density for each controller."""

    controllers = [
        controller
        for controller in CONTROLLER_ORDER
        if (controller, condition.value) in results.visitation
    ]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7.7),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for ax, controller in zip(axes.flat, controllers, strict=False):
        counts = results.visitation[(controller, condition.value)]
        normalized = counts / max(float(counts.sum()), 1.0)
        image = ax.imshow(
            np.log1p(10_000 * normalized).T,
            origin="lower",
            extent=(
                results.x_edges[0],
                results.x_edges[-1],
                results.y_edges[0],
                results.y_edges[-1],
            ),
            cmap="magma",
            aspect="equal",
            vmin=0,
        )
        _draw_world(ax, config)
        ax.set_title(CONTROLLER_LABELS[controller])
    for unused_ax in axes.flat[len(controllers) :]:
        unused_ax.set_visible(False)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="Log-scaled fraction of recorded positions",
            shrink=0.82,
        )
    figure.suptitle(
        f"Visitation density: {condition.value.replace('_', ' ')}",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_odor_blind_comparison(
    summaries: Sequence[AblationSummary],
    comparisons: Sequence[PairedAblationComparison],
    output_path: Path,
) -> None:
    """Plot the inference-only odor ablation with paired-test statistics."""

    if not summaries or not comparisons:
        raise ValueError("Ablation summaries and paired comparisons are required.")
    policies = [
        controller
        for controller in (
            ControllerName.PPO_NORMAL.value,
            ControllerName.PPO_ROBUST.value,
        )
        if any(row.policy == controller for row in summaries)
    ]
    conditions = (
        ObservationCondition.UNMASKED.value,
        ObservationCondition.ODOR_BLIND.value,
    )
    labels = {
        ObservationCondition.UNMASKED.value: "Full observation",
        ObservationCondition.ODOR_BLIND.value: "Odor hidden",
    }
    colors = {
        ObservationCondition.UNMASKED.value: "#2563a8",
        ObservationCondition.ODOR_BLIND.value: "#dc4331",
    }
    lookup = {(row.policy, row.observation_condition): row for row in summaries}
    comparison_lookup = {row.policy: row for row in comparisons}
    x = np.arange(len(policies), dtype=np.float64)
    width = 0.34
    figure, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    for index, condition in enumerate(conditions):
        rows = [lookup[(policy, condition)] for policy in policies]
        values = [row.success_rate for row in rows]
        errors = np.asarray(
            [
                [max(0.0, row.success_rate - row.success_ci95_low) for row in rows],
                [max(0.0, row.success_ci95_high - row.success_rate) for row in rows],
            ],
            dtype=np.float64,
        )
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=labels[condition],
            color=colors[condition],
            yerr=errors,
            capsize=4,
            error_kw={"linewidth": 1.0},
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.0f}%" for value in values],
            padding=4,
            fontsize=9,
        )
    for policy_index, policy in enumerate(policies):
        comparison = comparison_lookup[policy]
        p_value = comparison.exact_mcnemar_p_value
        p_label = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        ax.text(
            policy_index,
            106,
            f"paired Δ {comparison.success_rate_change_percentage_points:+.0f} pp\n{p_label}",
            ha="center",
            va="top",
            fontsize=8,
            color="#334155",
        )
    ax.set_xticks(x, [CONTROLLER_LABELS[policy] for policy in policies])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Successful episodes (%)")
    ax.set_xlabel(
        "Physical plume and sensors remain active; only odor-derived policy inputs are masked.",
        labelpad=12,
        fontsize=8,
        color="#475569",
    )
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncols=2)
    episodes = {row.episodes for row in summaries}
    sample_note = f"n={next(iter(episodes))} paired seeds" if len(episodes) == 1 else ""
    figure.suptitle(
        f"Does PPO need odor? Inference-only ablation ({sample_note})",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_cue_ablation_comparison(
    summaries: Sequence[AblationSummary],
    output_path: Path,
) -> None:
    """Plot full, odor-hidden, wind-hidden, and combined policy inputs."""

    if not summaries:
        raise ValueError("Ablation summaries are required.")
    policies = (
        ControllerName.PPO_NORMAL.value,
        ControllerName.PPO_ROBUST.value,
    )
    conditions = (
        ObservationCondition.UNMASKED.value,
        ObservationCondition.ODOR_BLIND.value,
        ObservationCondition.WIND_BLIND.value,
        ObservationCondition.ODOR_WIND_BLIND.value,
    )
    labels = {
        ObservationCondition.UNMASKED.value: "Full",
        ObservationCondition.ODOR_BLIND.value: "Odor\nhidden",
        ObservationCondition.WIND_BLIND.value: "Wind\nhidden",
        ObservationCondition.ODOR_WIND_BLIND.value: "Both\nhidden",
    }
    colors = {
        ObservationCondition.UNMASKED.value: "#2563a8",
        ObservationCondition.ODOR_BLIND.value: "#dc4331",
        ObservationCondition.WIND_BLIND.value: "#7c3aed",
        ObservationCondition.ODOR_WIND_BLIND.value: "#64748b",
    }
    lookup = {(row.policy, row.observation_condition): row for row in summaries}
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.4),
        constrained_layout=True,
        sharey=True,
    )
    x = np.arange(len(conditions), dtype=np.float64)
    for ax, policy in zip(axes, policies, strict=True):
        rows = [lookup[(policy, condition)] for condition in conditions]
        values = [row.success_rate for row in rows]
        errors = np.asarray(
            [
                [max(0.0, row.success_rate - row.success_ci95_low) for row in rows],
                [max(0.0, row.success_ci95_high - row.success_rate) for row in rows],
            ],
            dtype=np.float64,
        )
        bars = ax.bar(
            x,
            values,
            width=0.68,
            color=[colors[condition] for condition in conditions],
            yerr=errors,
            capsize=4,
            error_kw={"linewidth": 1.0},
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.0f}%" for value in values],
            padding=4,
            fontsize=9,
        )
        ax.set_xticks(x, [labels[condition] for condition in conditions])
        ax.set_title(CONTROLLER_LABELS[policy], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Successful episodes (%)")
    episodes = {row.episodes for row in summaries}
    sample_note = f"n={next(iter(episodes))} paired seeds" if len(episodes) == 1 else ""
    figure.suptitle(
        f"Which cue carries search? Inference-only ablations ({sample_note})",
        fontsize=14,
        fontweight="bold",
    )
    figure.supxlabel(
        "Physical plume and world remain unchanged; only named policy inputs are masked.",
        fontsize=8,
        color="#475569",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_wind_validity_comparison(
    summaries: Sequence[AblationSummary],
    output_path: Path,
) -> None:
    """Plot correct, zeroed, and valid-but-rotated wind under two odor states."""

    if not summaries:
        raise ValueError("Ablation summaries are required.")
    policies = (
        ControllerName.PPO_NORMAL.value,
        ControllerName.PPO_ROBUST.value,
    )
    condition_grid = (
        (
            ObservationCondition.UNMASKED.value,
            ObservationCondition.WIND_BLIND.value,
            ObservationCondition.WIND_ROTATED.value,
        ),
        (
            ObservationCondition.ODOR_BLIND.value,
            ObservationCondition.ODOR_WIND_BLIND.value,
            ObservationCondition.ODOR_BLIND_WIND_ROTATED.value,
        ),
    )
    wind_labels = ("Correct wind", "Zero vector", "Wind ±90°")
    colors = ("#2563a8", "#64748b", "#e76f51")
    lookup = {(row.policy, row.observation_condition): row for row in summaries}
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.5),
        constrained_layout=True,
        sharey=True,
    )
    centers = np.arange(2, dtype=np.float64)
    width = 0.23
    for ax, policy in zip(axes, policies, strict=True):
        for wind_index, (wind_label, color) in enumerate(zip(wind_labels, colors, strict=True)):
            rows = [
                lookup[(policy, condition_grid[odor_index][wind_index])] for odor_index in range(2)
            ]
            values = [row.success_rate for row in rows]
            errors = np.asarray(
                [
                    [max(0.0, row.success_rate - row.success_ci95_low) for row in rows],
                    [max(0.0, row.success_ci95_high - row.success_rate) for row in rows],
                ],
                dtype=np.float64,
            )
            x = centers + (wind_index - 1) * width
            bars = ax.bar(
                x,
                values,
                width=width,
                color=color,
                label=wind_label,
                yerr=errors,
                capsize=3,
                error_kw={"linewidth": 0.9},
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.0f}%" for value in values],
                padding=4,
                fontsize=8,
            )
        ax.set_xticks(centers, ("Odor available", "Odor hidden"))
        ax.set_title(CONTROLLER_LABELS[policy], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Successful episodes (%)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    episodes = {row.episodes for row in summaries}
    sample_note = f"n={next(iter(episodes))} paired seeds" if len(episodes) == 1 else ""
    figure.suptitle(
        f"Does wind direction matter? Valid-vector control ({sample_note})",
        fontsize=14,
        fontweight="bold",
    )
    figure.supxlabel(
        "Physical world unchanged; ±90° wind remains unit length and fixed within each episode.",
        fontsize=8,
        color="#475569",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_geometry_shift_comparison(
    summaries: Sequence[GeometrySummary],
    output_path: Path,
) -> None:
    """Plot aligned and crosswind-decoupled success under two input states."""

    if not summaries:
        raise ValueError("Geometry summaries are required.")
    policies = (
        ControllerName.PPO_NORMAL.value,
        ControllerName.PPO_ROBUST.value,
    )
    observations = (
        ObservationCondition.UNMASKED.value,
        ObservationCondition.ODOR_BLIND.value,
    )
    geometries = (
        GeometryCondition.SHIFTED_ALIGNED.value,
        GeometryCondition.CROSSWIND_DECOUPLED.value,
    )
    labels = ("Shifted but aligned", "Opposite crosswind lane")
    colors = ("#2563a8", "#e76f51")
    lookup = {
        (row.policy, row.observation_condition, row.geometry_condition): row for row in summaries
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.4),
        constrained_layout=True,
        sharey=True,
    )
    centers = np.arange(len(observations), dtype=np.float64)
    width = 0.34
    for ax, policy in zip(axes, policies, strict=True):
        for geometry_index, (geometry, label, color) in enumerate(
            zip(geometries, labels, colors, strict=True)
        ):
            rows = [lookup[(policy, observation, geometry)] for observation in observations]
            values = [row.success_rate for row in rows]
            errors = np.asarray(
                [
                    [max(0.0, row.success_rate - row.success_ci95_low) for row in rows],
                    [max(0.0, row.success_ci95_high - row.success_rate) for row in rows],
                ],
                dtype=np.float64,
            )
            x = centers + (geometry_index - 0.5) * width
            bars = ax.bar(
                x,
                values,
                width=width,
                color=color,
                label=label,
                yerr=errors,
                capsize=4,
                error_kw={"linewidth": 1.0},
            )
            ax.bar_label(
                bars,
                labels=[f"{value:.0f}%" for value in values],
                padding=4,
                fontsize=9,
            )
        ax.set_xticks(centers, ("Full observation", "Odor hidden"))
        ax.set_title(CONTROLLER_LABELS[policy], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Successful episodes (%)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    episodes = {row.episodes for row in summaries}
    sample_note = f"n={next(iter(episodes))} paired seeds" if len(episodes) == 1 else ""
    figure.suptitle(
        f"Does search survive opposite source/start lanes? ({sample_note})",
        fontsize=13.5,
        fontweight="bold",
    )
    figure.supxlabel(
        "Frozen policies; open world; same lane marginals, aligned versus opposite relationships.",
        fontsize=8,
        color="#475569",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_geometry_design_schematic(output_path: Path) -> None:
    """Draw the aligned-versus-decoupled source/start geometry manipulation."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    conditions = (
        ("Shifted but aligned", 3.0),
        ("Crosswind decoupled", 9.0),
    )
    for ax, (title, start_y) in zip(axes, conditions, strict=True):
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 12)
        ax.set_aspect("equal")
        ax.set_facecolor("#f8fafc")
        ax.axhline(3.0, color="#94a3b8", linestyle="--", linewidth=1.0)
        ax.axhline(9.0, color="#94a3b8", linestyle="--", linewidth=1.0)
        ax.scatter(2.5, 3.0, s=180, color="#dc4331", edgecolor="#7f1d1d", zorder=5)
        ax.scatter(
            16.5,
            start_y,
            s=180,
            marker="<",
            color="#2563a8",
            edgecolor="#1e3a8a",
            zorder=5,
        )
        ax.arrow(
            1.0,
            10.8,
            2.0,
            0.0,
            width=0.035,
            head_width=0.35,
            head_length=0.45,
            color="#0e7490",
            length_includes_head=True,
        )
        ax.annotate("wind", (1.0, 10.2), color="#0e7490", fontsize=9)
        ax.annotate("source", (2.5, 3.0), xytext=(0, -20), textcoords="offset points", ha="center")
        ax.annotate(
            "start",
            (16.5, start_y),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
        )
        ax.annotate(
            "upwind route",
            xy=(3.5, start_y),
            xytext=(15.5, start_y),
            arrowprops={"arrowstyle": "->", "color": "#475569", "linestyle": ":"},
            ha="right",
            va="bottom",
            color="#475569",
            fontsize=9,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Downwind distance")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Crosswind position")
    figure.suptitle(
        "Phase 5.4 manipulation: reverse source/start lane alignment",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_geometry_trajectory_audit(
    trajectories: Sequence[GeometryTrajectory],
    output_path: Path,
) -> None:
    """Plot Robust PPO odor-blind paths by geometry and source lane."""

    if not trajectories:
        raise ValueError("Geometry trajectories are required.")
    geometries = (
        GeometryCondition.SHIFTED_ALIGNED.value,
        GeometryCondition.CROSSWIND_DECOUPLED.value,
    )
    geometry_labels = ("Shifted but aligned", "Opposite start lane")
    source_lanes = (3.0, 9.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 7.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for row_index, (geometry, geometry_label) in enumerate(
        zip(geometries, geometry_labels, strict=True)
    ):
        for column_index, source_y in enumerate(source_lanes):
            ax = axes[row_index, column_index]
            rows = [
                row
                for row in trajectories
                if row.geometry_condition == geometry and row.source_position[1] == source_y
            ]
            for success in (False, True):
                for row in (item for item in rows if item.success is success):
                    ax.plot(
                        row.trajectory[:, 0],
                        row.trajectory[:, 1],
                        color="#16a34a" if success else "#94a3b8",
                        alpha=0.55 if success else 0.22,
                        linewidth=1.0,
                    )
            ax.scatter(
                [row.start_position[0] for row in rows],
                [row.start_position[1] for row in rows],
                s=12,
                color="#2563a8",
                alpha=0.65,
                zorder=4,
            )
            ax.scatter(
                2.5,
                source_y,
                s=95,
                color="#dc4331",
                edgecolor="#7f1d1d",
                zorder=5,
            )
            ax.axhline(6.0, color="#cbd5e1", linestyle="--", linewidth=0.8)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 12)
            ax.set_aspect("equal")
            ax.set_facecolor("#f8fafc")
            successes = sum(row.success for row in rows)
            ax.set_title(
                f"{geometry_label} · source y={source_y:.0f}\n{successes}/{len(rows)} successes",
                fontsize=10,
                fontweight="bold",
            )
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Downwind position x")
    for ax in axes[:, 0]:
        ax.set_ylabel("Crosswind position y")
    figure.legend(
        handles=(
            Line2D([0], [0], color="#16a34a", label="Successful path"),
            Line2D([0], [0], color="#94a3b8", label="Failed path"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#2563a8",
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#dc4331",
                label="Source",
            ),
        ),
        loc="outside right upper",
        ncols=1,
        frameon=False,
    )
    figure.suptitle(
        "Robust PPO without odor sweeps across the arena",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
