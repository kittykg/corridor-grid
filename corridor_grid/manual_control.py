import argparse
from typing import Any

import pygame
import gymnasium as gym
from gymnasium import Env

import corridor_grid
from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv
from corridor_grid.envs.door_corridor import DoorCorridorEnv, DoorCorridorAction


class ManualControl:
    env: Env
    seed: int | None
    closed: bool

    def __init__(self, env: Env, seed: int | None = None) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: int):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event: pygame.event.Event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            self.closed = True
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = self._get_key_to_action_map()
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    @staticmethod
    def _get_key_to_action_map() -> dict[str, Any]:
        raise NotImplementedError


class SpecialStateCorridorManualControl(ManualControl):
    env: BaseSpecialStateCorridorEnv

    def __init__(
        self, env: BaseSpecialStateCorridorEnv, seed: int | None = None
    ) -> None:
        super().__init__(env, seed=seed)

    @staticmethod
    def _get_key_to_action_map() -> dict[str, Any]:
        return {
            "left": 0,
            "right": 1,
        }


class DoorCorridorManualControl(ManualControl):
    env: DoorCorridorEnv

    def __init__(self, env: DoorCorridorEnv, seed: int | None = None) -> None:
        super().__init__(env, seed=seed)

    @staticmethod
    def _get_key_to_action_map() -> dict[str, Any]:
        return {
            "left": DoorCorridorAction.LEFT,
            "right": DoorCorridorAction.RIGHT,
            "up": DoorCorridorAction.FORWARD,
            "space": DoorCorridorAction.TOGGLE,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        default="CG-SC-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="human")
    if "CG-DC" in args.env_id:
        mc_class = DoorCorridorManualControl
    else:
        mc_class = SpecialStateCorridorManualControl

    manual_control = mc_class(env, seed=args.seed)  # type: ignore
    manual_control.start()
