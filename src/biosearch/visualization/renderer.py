"""Pygame renderer for the 2D simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from biosearch.environment import BioSearchSimulation


@dataclass(frozen=True)
class RenderStyle:
    """Colors and dimensions for the interactive view."""

    window_size: tuple[int, int] = (1_180, 700)
    panel_width: int = 280
    margin: int = 28
    background: tuple[int, int, int] = (244, 247, 250)
    world_background: tuple[int, int, int] = (252, 253, 255)
    ink: tuple[int, int, int] = (32, 43, 56)
    muted: tuple[int, int, int] = (101, 116, 139)
    obstacle: tuple[int, int, int] = (82, 96, 109)
    trajectory: tuple[int, int, int] = (66, 109, 164)
    agent: tuple[int, int, int] = (26, 71, 118)
    source: tuple[int, int, int] = (220, 67, 49)
    sensor: tuple[int, int, int] = (255, 183, 3)


class PygameRenderer:
    """Draw the complete simulation state in an interactive window."""

    def __init__(
        self,
        simulation: BioSearchSimulation,
        *,
        caption: str = "BioSearch RL",
        style: RenderStyle | None = None,
    ) -> None:
        pygame.init()
        pygame.font.init()
        self.simulation = simulation
        self.style = style or RenderStyle()
        self.screen = pygame.display.set_mode(self.style.window_size)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)
        self.small_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 34)

        usable_width = self.style.window_size[0] - self.style.panel_width - 3 * self.style.margin
        usable_height = self.style.window_size[1] - 2 * self.style.margin
        world = simulation.config.world
        self.scale = min(usable_width / world.width, usable_height / world.height)
        drawn_height = world.height * self.scale
        self.origin_x = float(self.style.margin)
        self.origin_y = (self.style.window_size[1] + drawn_height) / 2
        self.world_rect = pygame.Rect(
            round(self.origin_x),
            round(self.origin_y - drawn_height),
            round(world.width * self.scale),
            round(drawn_height),
        )

    def _point(self, world_point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        return (
            round(self.origin_x + float(world_point[0]) * self.scale),
            round(self.origin_y - float(world_point[1]) * self.scale),
        )

    def _draw_world(self) -> None:
        sim = self.simulation
        style = self.style
        pygame.draw.rect(self.screen, style.world_background, self.world_rect)
        pygame.draw.rect(self.screen, (203, 213, 225), self.world_rect, width=2)

        plume = sim.plume.snapshot()
        if len(plume.positions):
            visible_scale = max(sim.config.plume.initial_intensity, 1e-9)
            for position, intensity in zip(
                plume.positions,
                plume.intensities,
                strict=True,
            ):
                if not (
                    0 <= position[0] <= sim.config.world.width
                    and 0 <= position[1] <= sim.config.world.height
                ):
                    continue
                strength = float(np.clip(intensity / visible_scale, 0.0, 1.0))
                color = (
                    150 - round(40 * strength),
                    205 - round(35 * strength),
                    235,
                )
                radius = max(2, round(2.5 + 3.5 * strength))
                pygame.draw.circle(self.screen, color, self._point(position), radius)

        for obstacle in sim.config.world.obstacles:
            top_left = self._point((obstacle.x, obstacle.y + obstacle.height))
            obstacle_rect = pygame.Rect(
                top_left[0],
                top_left[1],
                round(obstacle.width * self.scale),
                round(obstacle.height * self.scale),
            )
            pygame.draw.rect(self.screen, style.obstacle, obstacle_rect, border_radius=3)
            pygame.draw.rect(self.screen, (55, 65, 81), obstacle_rect, width=2, border_radius=3)

        if len(sim.trajectory) >= 2:
            points = [self._point(point) for point in sim.trajectory]
            pygame.draw.lines(self.screen, style.trajectory, False, points, width=2)
            # Detection markers are display-only and do not affect the simulation.
            for index, (point, detected) in enumerate(
                zip(sim.trajectory, sim.detection_history, strict=True)
            ):
                if detected and index % 3 == 0:
                    pygame.draw.circle(self.screen, style.sensor, self._point(point), 3)

        source_center = self._point(sim.config.world.source_position)
        source_radius = max(4, round(sim.config.world.source_radius * self.scale))
        pygame.draw.circle(self.screen, (254, 226, 226), source_center, source_radius + 5)
        pygame.draw.circle(self.screen, style.source, source_center, source_radius)
        pygame.draw.circle(self.screen, (127, 29, 29), source_center, source_radius, width=2)
        source_label = self.small_font.render("source", True, (127, 29, 29))
        self.screen.blit(
            source_label,
            (source_center[0] - source_label.get_width() // 2, source_center[1] + 25),
        )

        reading = sim.last_sensor_reading
        for position, value in (
            (reading.left_position, reading.left),
            (reading.right_position, reading.right),
        ):
            sensor_center = self._point(position)
            sensor_radius = 4 + round(5 * value / sim.config.sensors.max_reading)
            pygame.draw.line(
                self.screen,
                (148, 163, 184),
                self._point(sim.position),
                sensor_center,
                width=1,
            )
            pygame.draw.circle(self.screen, style.sensor, sensor_center, sensor_radius)
            pygame.draw.circle(self.screen, (146, 64, 14), sensor_center, sensor_radius, width=1)

        self._draw_agent()
        self._draw_wind()

    def _draw_agent(self) -> None:
        sim = self.simulation
        center = self._point(sim.position)
        radius = max(8, round(sim.config.agent.radius * self.scale))
        heading = np.array([np.cos(sim.heading), np.sin(sim.heading)])
        left_axis = np.array([-heading[1], heading[0]])
        nose = sim.position + heading * sim.config.agent.radius * 1.45
        rear_left = (
            sim.position
            - heading * sim.config.agent.radius * 0.85
            + left_axis * sim.config.agent.radius
        )
        rear_right = (
            sim.position
            - heading * sim.config.agent.radius * 0.85
            - left_axis * sim.config.agent.radius
        )
        pygame.draw.circle(self.screen, (219, 234, 254), center, radius + 4)
        pygame.draw.polygon(
            self.screen,
            self.style.agent,
            [self._point(nose), self._point(rear_left), self._point(rear_right)],
        )
        pygame.draw.polygon(
            self.screen,
            (15, 47, 87),
            [self._point(nose), self._point(rear_left), self._point(rear_right)],
            width=2,
        )

    def _draw_wind(self) -> None:
        world = self.simulation.config.world
        direction = np.array([np.cos(world.wind_direction), np.sin(world.wind_direction)])
        start = np.array([0.9, world.height - 0.8])
        end = start + 1.4 * direction
        start_px, end_px = self._point(start), self._point(end)
        pygame.draw.line(self.screen, (14, 116, 144), start_px, end_px, width=4)

        angle = np.arctan2(direction[1], direction[0])
        for offset in (0.55, -0.55):
            wing = end - 0.35 * np.array([np.cos(angle + offset), np.sin(angle + offset)])
            pygame.draw.line(self.screen, (14, 116, 144), end_px, self._point(wing), width=4)
        label = self.small_font.render("wind", True, (14, 116, 144))
        self.screen.blit(label, (start_px[0], start_px[1] + 8))

    def _draw_panel(self, controller_state: str) -> None:
        sim = self.simulation
        style = self.style
        panel_x = style.window_size[0] - style.panel_width
        pygame.draw.rect(
            self.screen,
            (236, 241, 247),
            pygame.Rect(panel_x, 0, style.panel_width, style.window_size[1]),
        )
        pygame.draw.line(
            self.screen,
            (203, 213, 225),
            (panel_x, 0),
            (panel_x, style.window_size[1]),
            width=2,
        )

        x, y = panel_x + 24, 28
        self.screen.blit(self.title_font.render("BioSearch RL", True, style.ink), (x, y))
        y += 44
        subtitle = self.small_font.render("Abstract odor-source search", True, style.muted)
        self.screen.blit(subtitle, (x, y))
        y += 46

        reading = sim.last_sensor_reading
        rows = (
            ("Controller", controller_state),
            ("Step", f"{sim.step_count} / {sim.config.agent.max_steps}"),
            ("Action", sim.previous_action.name),
            ("Left sensor", f"{reading.left:.3f}"),
            ("Right sensor", f"{reading.right:.3f}"),
            ("Odor detected", "YES" if reading.detected else "no"),
            ("Collisions", str(sim.collision_count)),
            ("Puffs", str(len(sim.plume.positions))),
        )
        for label, value in rows:
            self.screen.blit(self.small_font.render(label, True, style.muted), (x, y))
            self.screen.blit(self.font.render(value, True, style.ink), (x, y + 18))
            y += 46

        y += 8
        controls = (
            "Manual controls",
            "W / Up   forward",
            "A,D / Left,Right   turn + move",
            "Q,E   sharp turn",
            "Space   remain still",
            "R   reset    Esc   quit",
        )
        for index, line in enumerate(controls):
            color = style.ink if index == 0 else style.muted
            font = self.font if index == 0 else self.small_font
            self.screen.blit(font.render(line, True, color), (x, y))
            y += 26

        if sim.success or sim.truncated:
            message = "SOURCE FOUND" if sim.success else "TIME LIMIT"
            color = (21, 128, 61) if sim.success else (180, 83, 9)
            banner = self.font.render(message, True, (255, 255, 255))
            banner_rect = pygame.Rect(panel_x + 20, style.window_size[1] - 58, 240, 38)
            pygame.draw.rect(self.screen, color, banner_rect, border_radius=6)
            self.screen.blit(banner, banner.get_rect(center=banner_rect.center))

    def render(self, *, controller_state: str) -> None:
        """Draw one frame and present it."""

        self.screen.fill(self.style.background)
        self._draw_world()
        self._draw_panel(controller_state)
        pygame.display.flip()

    def capture_frame(
        self,
        *,
        size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Copy the current screen into an ``(height, width, RGB)`` array."""

        surface = self.screen
        if size is not None:
            surface = pygame.transform.smoothscale(surface, size)
        frame = pygame.surfarray.array3d(surface)
        return np.transpose(frame, (1, 0, 2)).copy()

    def tick(self, fps: int) -> None:
        """Limit interactive playback speed."""

        self.clock.tick(fps)

    @staticmethod
    def close() -> None:
        """Release the Pygame window."""

        pygame.quit()
