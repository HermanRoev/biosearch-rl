"""Command-line entry point for interactive controller demos."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from biosearch.controllers import MothController, RandomController
from biosearch.environment import Action, BioSearchSimulation
from biosearch.evaluation import action_for_controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BioSearch RL abstract plume simulation.")
    parser.add_argument(
        "--controller",
        choices=("random", "moth", "manual"),
        default="moth",
        help="Random baseline, moth-inspired baseline, or keyboard control.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Simulation/controller seed.")
    parser.add_argument("--steps", type=int, default=1_200, help="Maximum demo steps.")
    parser.add_argument("--fps", type=int, default=30, help="Rendered frames per second.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a window (automatic controllers only).",
    )
    return parser


def _manual_action(pygame_module: object) -> Action:
    pygame = pygame_module
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return Action.FORWARD
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return Action.TURN_LEFT
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return Action.TURN_RIGHT
    if keys[pygame.K_q]:
        return Action.SHARP_LEFT
    if keys[pygame.K_e]:
        return Action.SHARP_RIGHT
    return Action.STILL


def _summary(simulation: BioSearchSimulation, controller_name: str) -> str:
    return (
        f"controller={controller_name} seed={simulation.seed} "
        f"steps={simulation.step_count} success={simulation.success} "
        f"collisions={simulation.collision_count} "
        f"final_position=({simulation.position[0]:.2f}, {simulation.position[1]:.2f}) "
        f"puffs={len(simulation.plume.positions)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run an interactive or headless demo."""

    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.headless and args.controller == "manual":
        raise SystemExit("Manual control requires a rendered window.")

    simulation = BioSearchSimulation(seed=args.seed)
    controller: RandomController | MothController = (
        MothController() if args.controller == "moth" else RandomController(seed=args.seed)
    )

    if args.headless:
        while (
            simulation.step_count < args.steps
            and not simulation.terminated
            and not simulation.truncated
        ):
            simulation.step(action_for_controller(controller, simulation))
        print(_summary(simulation, args.controller))
        return 0

    # Import Pygame only for rendered runs so headless physics users do not need
    # a video device. SDL can still be set to "dummy" by CI before this point.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    from biosearch.visualization import PygameRenderer

    renderer = PygameRenderer(simulation)
    running = True
    try:
        while running:
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_requested = True

            if not running:
                break
            if reset_requested:
                simulation.reset(seed=args.seed)
                controller.reset(seed=args.seed)

            finished = simulation.terminated or simulation.truncated
            hit_demo_limit = simulation.step_count >= args.steps
            if not finished and not hit_demo_limit:
                if args.controller == "manual":
                    action = _manual_action(pygame)
                    controller_state = "MANUAL"
                else:
                    action = action_for_controller(controller, simulation)
                    controller_state = controller.state_name
                simulation.step(action)
            else:
                controller_state = args.controller.upper()

            renderer.render(controller_state=controller_state)
            renderer.tick(args.fps)
    finally:
        renderer.close()

    print(_summary(simulation, args.controller))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
