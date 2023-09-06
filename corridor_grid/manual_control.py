import argparse

import pygame
import gymnasium as gym
from gymnasium import Env

import corridor_grid
from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv


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
        raise NotImplementedError


class SpecialStateCorridorManualControl(ManualControl):
    env: BaseSpecialStateCorridorEnv

    def __init__(
        self, env: BaseSpecialStateCorridorEnv, seed: int | None = None
    ) -> None:
        super().__init__(env, seed=seed)

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

        key_to_action = {
            "left": 0,
            "right": 1,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


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

    env: BaseSpecialStateCorridorEnv = gym.make(
        args.env_id, render_mode="human"
    )  # type: ignore
    manual_control = SpecialStateCorridorManualControl(env, seed=args.seed)
    manual_control.start()
