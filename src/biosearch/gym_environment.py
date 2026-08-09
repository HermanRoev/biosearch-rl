"""Gymnasium interface for reinforcement-learning experiments.

The policy observation contains local sensing and internal memory only. Source
coordinates and exact source distance are used by the simulator and optional
reward shaping, but never appear in the policy observation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import pi
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from biosearch.config import Rectangle, SimulationConfig
from biosearch.environment import Action, BioSearchSimulation

FloatArray = NDArray[np.float64]
Observation = NDArray[np.float32]
ConfigSampler = Callable[[np.random.Generator, SimulationConfig], SimulationConfig]


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms used during training.

    ``distance_progress`` uses hidden simulator state only to compute the
    training signal. Distance is never included in :attr:`observation_names`.
    """

    success: float = 50.0
    step_penalty: float = -0.01
    collision: float = -0.35
    odor_detection: float = 0.01
    distance_progress: float = 0.60


class BioSearchEnv(gym.Env[Observation, int]):
    """Gymnasium wrapper around :class:`BioSearchSimulation`."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    observation_names = (
        "left_odor",
        "right_odor",
        "odor_difference",
        "odor_detected",
        "time_since_detection",
        "wind_heading_sin",
        "wind_heading_cos",
        "previous_action",
        "left_odor_moving_average",
        "right_odor_moving_average",
        "front_obstacle_proximity",
        "left_obstacle_proximity",
        "right_obstacle_proximity",
    )

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
        randomize_start: bool = True,
        config_sampler: ConfigSampler | None = None,
        odor_history_length: int = 8,
        max_detection_age: int = 100,
        obstacle_sensor_range: float = 2.5,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode {render_mode!r}.")
        if odor_history_length <= 0 or max_detection_age <= 0:
            raise ValueError("Observation history settings must be positive.")
        if obstacle_sensor_range <= 0:
            raise ValueError("obstacle_sensor_range must be positive.")

        self.base_config = config or SimulationConfig()
        self.config = self.base_config
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.randomize_start = randomize_start
        self.config_sampler = config_sampler
        self.odor_history_length = odor_history_length
        self.max_detection_age = max_detection_age
        self.obstacle_sensor_range = obstacle_sensor_range

        self.action_space = spaces.Discrete(len(Action))
        low = np.array(
            [0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        high = np.ones(len(self.observation_names), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.simulation = BioSearchSimulation(self.config, seed=0)
        self._odor_history: deque[tuple[float, float]] = deque(maxlen=self.odor_history_length)
        self._time_since_detection = self.max_detection_age
        self._previous_distance = self._distance_to_source()
        self._renderer: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset with Gymnasium-compliant seeding and an optional fixed pose."""

        super().reset(seed=seed)
        options = options or {}
        if self.config_sampler is not None:
            self.config = self.config_sampler(self.np_random, self.base_config)
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            self.simulation = BioSearchSimulation(self.config, seed=0)
        episode_seed = (
            int(options["simulation_seed"])
            if "simulation_seed" in options
            else int(self.np_random.integers(0, np.iinfo(np.uint32).max))
        )
        use_random_start = bool(options.get("randomize_start", self.randomize_start))
        sampled_position, sampled_heading = (
            self._sample_start_pose()
            if use_random_start
            else (
                self.config.world.agent_start,
                self.config.world.agent_start_heading,
            )
        )

        if "agent_position" in options:
            raw_position = options["agent_position"]
            agent_position = (float(raw_position[0]), float(raw_position[1]))
        else:
            agent_position = sampled_position

        heading = float(options["heading"]) if "heading" in options else sampled_heading

        reading = self.simulation.reset(
            seed=episode_seed,
            agent_position=agent_position,
            heading=heading,
        )
        self._odor_history.clear()
        self._odor_history.append((reading.left, reading.right))
        self._time_since_detection = 0 if reading.detected else self.max_detection_age
        self._previous_distance = self._distance_to_source()

        observation = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _sample_start_pose(self) -> tuple[tuple[float, float], float]:
        """Sample a downwind pose within the Milestone 2 evaluation corridor."""

        nominal_start = np.asarray(self.config.world.agent_start, dtype=np.float64)
        crosswind_offset = float(self.np_random.uniform(-2.0, 2.0))
        crosswind = np.array(
            [
                -np.sin(self.config.world.wind_direction),
                np.cos(self.config.world.wind_direction),
            ],
            dtype=np.float64,
        )
        position = nominal_start + crosswind_offset * crosswind
        margin = self.config.agent.radius + 0.1
        position[0] = np.clip(position[0], margin, self.config.world.width - margin)
        position[1] = np.clip(position[1], margin, self.config.world.height - margin)
        heading = float(
            self.config.world.agent_start_heading + self.np_random.uniform(-pi / 4, pi / 4)
        )
        return (float(position[0]), float(position[1])), heading

    def step(
        self,
        action: int,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one discrete action and calculate the shaped training reward."""

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}.")

        result = self.simulation.step(int(action))
        reading = result.sensor_reading
        self._odor_history.append((reading.left, reading.right))
        self._time_since_detection = (
            0 if reading.detected else min(self.max_detection_age, self._time_since_detection + 1)
        )

        new_distance = self._distance_to_source()
        progress = self._previous_distance - new_distance
        self._previous_distance = new_distance
        terms = {
            "step": self.reward_config.step_penalty,
            "collision": self.reward_config.collision if result.collision else 0.0,
            "odor_detection": (self.reward_config.odor_detection if reading.detected else 0.0),
            "distance_progress": self.reward_config.distance_progress * progress,
            "success": self.reward_config.success if result.success else 0.0,
        }
        reward = float(sum(terms.values()))

        observation = self._get_observation()
        info = self._get_info()
        info["reward_terms"] = terms
        if self.render_mode == "human":
            self.render()
        return (
            observation,
            reward,
            result.terminated,
            result.truncated,
            info,
        )

    def _get_observation(self) -> Observation:
        reading = self.simulation.last_sensor_reading
        history = np.asarray(self._odor_history, dtype=np.float64)
        moving_average = history.mean(axis=0)
        relative_heading = (self.simulation.heading - self.config.world.wind_direction + pi) % (
            2 * pi
        ) - pi
        front, left, right = self._obstacle_proximities()
        observation = np.array(
            [
                reading.left,
                reading.right,
                reading.left - reading.right,
                float(reading.detected),
                self._time_since_detection / self.max_detection_age,
                np.sin(relative_heading),
                np.cos(relative_heading),
                int(self.simulation.previous_action) / (len(Action) - 1),
                moving_average[0],
                moving_average[1],
                front,
                left,
                right,
            ],
            dtype=np.float32,
        )
        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "success": self.simulation.success,
            "is_success": self.simulation.success,
            "step_count": self.simulation.step_count,
            "collisions": self.simulation.collision_count,
            "odor_detected": self.simulation.last_sensor_reading.detected,
        }

    def _distance_to_source(self) -> float:
        source = np.asarray(self.config.world.source_position, dtype=np.float64)
        return float(np.linalg.norm(self.simulation.position - source))

    def _obstacle_proximities(self) -> tuple[float, float, float]:
        offsets = (0.0, pi / 3, -pi / 3)
        values = [
            1.0 - self._ray_distance(self.simulation.heading + offset) / self.obstacle_sensor_range
            for offset in offsets
        ]
        return float(values[0]), float(values[1]), float(values[2])

    def _ray_distance(self, angle: float) -> float:
        origin = self.simulation.position
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        radius = self.config.agent.radius
        world = self.config.world

        inner_min = np.array([radius, radius], dtype=np.float64)
        inner_max = np.array(
            [world.width - radius, world.height - radius],
            dtype=np.float64,
        )
        boundary_distance = self._distance_to_boundary(
            origin,
            direction,
            inner_min,
            inner_max,
        )
        distance = min(self.obstacle_sensor_range, boundary_distance)
        for obstacle in world.obstacles:
            obstacle_distance = self._distance_to_rectangle(
                origin,
                direction,
                obstacle,
                padding=radius,
            )
            distance = min(distance, obstacle_distance)
        return max(0.0, min(self.obstacle_sensor_range, distance))

    @staticmethod
    def _distance_to_boundary(
        origin: FloatArray,
        direction: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
    ) -> float:
        distances: list[float] = []
        for axis in range(2):
            if direction[axis] > 1e-12:
                distances.append(float((upper[axis] - origin[axis]) / direction[axis]))
            elif direction[axis] < -1e-12:
                distances.append(float((lower[axis] - origin[axis]) / direction[axis]))
        positive = [distance for distance in distances if distance >= 0]
        return min(positive, default=float("inf"))

    @staticmethod
    def _distance_to_rectangle(
        origin: FloatArray,
        direction: FloatArray,
        rectangle: Rectangle,
        *,
        padding: float,
    ) -> float:
        lower = np.array(
            [rectangle.x - padding, rectangle.y - padding],
            dtype=np.float64,
        )
        upper = np.array(
            [
                rectangle.x + rectangle.width + padding,
                rectangle.y + rectangle.height + padding,
            ],
            dtype=np.float64,
        )
        entry, exit_ = -float("inf"), float("inf")
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                    return float("inf")
                continue
            first = (lower[axis] - origin[axis]) / direction[axis]
            second = (upper[axis] - origin[axis]) / direction[axis]
            near, far = sorted((float(first), float(second)))
            entry = max(entry, near)
            exit_ = min(exit_, far)
            if entry > exit_:
                return float("inf")
        if exit_ < 0:
            return float("inf")
        return max(0.0, entry)

    def render(self) -> Observation | None:
        """Render interactively or return an RGB array, according to render mode."""

        if self.render_mode is None:
            return None
        if self._renderer is None:
            from biosearch.visualization.renderer import PygameRenderer

            self._renderer = PygameRenderer(
                self.simulation,
                caption="BioSearch RL: PPO",
            )
        self._renderer.render(controller_state="RL POLICY")
        if self.render_mode == "human":
            self._renderer.tick(self.metadata["render_fps"])
            return None
        return self._renderer.capture_frame()

    def close(self) -> None:
        """Release renderer resources."""

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
