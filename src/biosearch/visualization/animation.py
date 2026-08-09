"""Animated GIF generation using the same Pygame renderer as the demo."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

from biosearch.config import SimulationConfig
from biosearch.environment import BioSearchSimulation
from biosearch.evaluation import (
    ControllerName,
    SensorCondition,
    action_for_controller,
    config_for_condition,
    make_controller,
)
from biosearch.visualization.renderer import PygameRenderer


def save_episode_gif(
    controller_name: ControllerName | str,
    condition: SensorCondition | str,
    seed: int,
    output_path: Path,
    *,
    base_config: SimulationConfig | None = None,
    frame_stride: int = 8,
    fps: int = 15,
    frame_size: tuple[int, int] = (826, 490),
) -> None:
    """Run an episode and stream rendered frames into a looping GIF."""

    if frame_stride <= 0 or fps <= 0:
        raise ValueError("frame_stride and fps must be positive.")
    selected_controller = ControllerName(controller_name)
    selected_condition = SensorCondition(condition)
    config = config_for_condition(selected_condition, base_config)
    simulation = BioSearchSimulation(config, seed=seed)
    controller = make_controller(selected_controller, seed=seed)
    renderer = PygameRenderer(
        simulation,
        caption=f"{selected_controller.title()}: {selected_condition}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            renderer.render(controller_state=controller.state_name)
            writer.append_data(renderer.capture_frame(size=frame_size))
            while not simulation.terminated and not simulation.truncated:
                action = action_for_controller(controller, simulation)
                simulation.step(action)
                final_step = simulation.terminated or simulation.truncated
                if simulation.step_count % frame_stride == 0 or final_step:
                    renderer.render(controller_state=controller.state_name)
                    writer.append_data(renderer.capture_frame(size=frame_size))
    finally:
        renderer.close()
