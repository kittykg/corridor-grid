from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces
import pygame


# Default environment configuration
REWARD_PER_STEP = -1
DEFAULT_START_STATE = 0
DEFAULT_TRUNCATE_TOLERANCE = 50

# Base environment (length 4) configuration
BASE_SMALL_CORRIDOR_ENV_SIZE = 4
BASE_SMALL_CORRIDOR_ENV_SPECIAL_STATE = 1

# Environment space types
ActionIntType = int
ActionNpIntType = np.int64
ActionStrType = str
ObservationType = dict[str, int | npt.NDArray[np.int64]]

# Pygame configuration
RENDER_FPS = 4
PYGAME_DEFAULT_SQUARE_SIZE = 100
PYGAME_DEFAULT_AGENT_RADIUS = PYGAME_DEFAULT_SQUARE_SIZE // 4
PYGAME_MAX_WINDOW_W = 10 * PYGAME_DEFAULT_SQUARE_SIZE
PYGAME_MAX_WINDOW_H = PYGAME_DEFAULT_SQUARE_SIZE
START_CELL_COLOUR: tuple[int, int, int, int] = (255, 128, 0, 160)
GOAL_CELL_COLOUR: tuple[int, int, int, int] = (0, 128, 255, 160)
SPECIAL_CELL_COLOUR: tuple[int, int, int, int] = (128, 0, 255, 160)
AGENT_COLOUR: tuple[int, int, int, int] = (0, 255, 0, 255)
LINE_COLOUR: tuple[int, int, int, int] = (0, 0, 0, 160)


@dataclass
class SpecialStateCorridorEnvCustomisationConfig:
    """
    A customisation config for the special state corridor environment.
    """

    corridor_length: int
    start_state: int | None
    goal_state: int
    special_states: list[int]
    truncate_tolerance: int

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any]
    ) -> "SpecialStateCorridorEnvCustomisationConfig":
        # Corridor length
        corridor_length = (
            config_dict["corridor_length"]
            if "corridor_length" in config_dict
            else BASE_SMALL_CORRIDOR_ENV_SIZE
        )
        assert (
            corridor_length >= 2
        ), "Corridor length must be at least 2 (a start and a goal)"

        # Start state, can be None (random start)
        start_state = (
            config_dict["start_state"] if "start_state" in config_dict else None
        )
        if start_state is not None:
            assert start_state >= 0, "Start state must be non-negative"
            assert (
                start_state < corridor_length
            ), "Start state must be within the corridor"

        # Goal state
        goal_state = (
            config_dict["goal_state"]
            if "goal_state" in config_dict
            else corridor_length - 1
        )
        assert goal_state >= 0, "Goal state must be non-negative"
        assert (
            goal_state < corridor_length
        ), "Goal state must be within the corridor"
        assert (
            goal_state != start_state
        ), "Goal state must be different from start state"

        # Special states
        special_states = (
            config_dict["special_states"]
            if "special_states" in config_dict
            else [BASE_SMALL_CORRIDOR_ENV_SPECIAL_STATE]
        )
        for s in special_states:
            assert s >= 0, f"Special state must be non-negative, but get {s}"
            assert (
                s < corridor_length
            ), f"Special state must be within the corridor, but get {s}"

        # Truncate tolerance
        truncate_tolerance = int(
            config_dict["truncate_tolerance"]
            if "truncate_tolerance" in config_dict
            else DEFAULT_TRUNCATE_TOLERANCE
        )
        assert truncate_tolerance > 0, "Truncate tolerance must be positive"

        return cls(
            corridor_length=corridor_length,
            start_state=start_state,
            goal_state=goal_state,
            special_states=special_states,
            truncate_tolerance=truncate_tolerance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "corridor_length": self.corridor_length,
            "start_state": self.start_state,
            "goal_state": self.goal_state,
            "special_states": self.special_states,
            "truncate_tolerance": self.truncate_tolerance,
        }

    def adjust_truncate_tolerance(
        self, new_truncate_tolerance: int
    ) -> "SpecialStateCorridorEnvCustomisationConfig":
        return replace(self, truncate_tolerance=new_truncate_tolerance)


class BaseSpecialStateCorridorEnv(gym.Env):
    """
    Base class for a special state corridor environments.
    A special state corridor is a corridor environment where the agent can only
    move left and right. The environment has a start and goal state, and there
    may be some special states where the action is reversed: if you move left,
    you end up going right and vice versa. But the agent's observation is only
    whether there is wall the same in every state, so the agent cannot tell if
    it is in a special state or not. The observation is a dict, however, will
    contain the agent location, if training with full observation.
    The default reward is -1 per step, and the episode terminates when the agent
    reaches the goal state. The episode could also be truncated if the steps
    exceed a certain limit controlled by `truncate_tolerance`.
    This class should not be imported directly, instead import the specific
    child classes `SmallLRCorridorEnv` and `LongLRCorridorEnv`.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": RENDER_FPS,
    }
    render_mode: Optional[str]

    ACTION = ["L", "R"]

    # Expected from the customisation config
    truncate_tolerance: int
    corridor_length: int
    start_state: int | None
    goal_state: int
    special_states: list[int]
    customisation_cfg: SpecialStateCorridorEnvCustomisationConfig

    action_space: spaces.Discrete
    observation_space: spaces.Dict

    _curr_start_state: int
    _agent_location: int
    _internal_step_counter: int

    window: Optional[pygame.Surface]
    clock: Optional[pygame.time.Clock]

    def __init__(
        self,
        render_mode: Optional[str] = None,
        customisation_cfg_dict: dict[str, Any] = dict(),
    ) -> None:
        super().__init__()

        if render_mode:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode: {render_mode}"
        self.render_mode = render_mode
        customisation_cfg = (
            SpecialStateCorridorEnvCustomisationConfig.from_dict(
                customisation_cfg_dict
            )
        )
        self.truncate_tolerance = customisation_cfg.truncate_tolerance
        self.corridor_length = customisation_cfg.corridor_length
        self.start_state = customisation_cfg.start_state
        self.goal_state = customisation_cfg.goal_state
        self.special_states = customisation_cfg.special_states
        self.customisation_cfg = customisation_cfg

        self.action_space = spaces.Discrete(len(self.ACTION))
        self.observation_space = spaces.Dict(
            {
                "wall_status": spaces.Box(
                    low=0, high=1, shape=(2,), dtype=np.int64
                ),
                "agent_location": spaces.Discrete(self.corridor_length),
            }
        )

        self.window = None
        self.clock = None

    # ---------- Main API methods ----------
    def step(
        self, action: ActionIntType | ActionNpIntType | ActionStrType
    ) -> tuple[ObservationType, float, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.
        If the agent reaches the goal state, the episode terminates.
        If the timestep limit is reached, the episode is truncated.
        Returns: observation, reward, terminated, truncated, info
        """
        self._agent_location = self._get_agent_new_location(action)

        terminated = self._agent_location == self.goal_state
        self._internal_step_counter += 1
        truncated = self._internal_step_counter >= self.truncate_tolerance
        reward = -1

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_agent_observation(),
            reward,
            terminated,
            truncated,
            {
                "action": self._get_action_str(action),
                "observation": self._get_agent_observation(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "distance_to_goal": self._get_distance_to_goal(),
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        if self.start_state is not None:
            # There is a fixed start
            self._agent_location = self.start_state
        else:
            # Random start
            super().reset(seed=seed)
            self._agent_location = np.random.choice(self.corridor_length)

        self._curr_start_state = self._agent_location
        self._internal_step_counter = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_agent_observation(), {
            "agent_location": self._agent_location,
            "goal": self.goal_state,
            "distance_to_goal": self._get_distance_to_goal(),
        }

    def render(self) -> str | npt.NDArray[np.uint8] | None:
        return self._render_frame()

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            pygame.quit()

    # ---------- Private methods ----------
    def _get_pygame_window_square_dimension(self) -> tuple[int, int, int]:
        # Return: (window_w, window_h, pygame_square_size)
        if (
            PYGAME_DEFAULT_SQUARE_SIZE * self.corridor_length
            > PYGAME_MAX_WINDOW_W
        ):
            pygame_square_size = PYGAME_MAX_WINDOW_W // self.corridor_length
        else:
            pygame_square_size = PYGAME_DEFAULT_SQUARE_SIZE
        window_w = pygame_square_size * self.corridor_length
        window_h = pygame_square_size
        return window_w, window_h, pygame_square_size

    def _render_frame(self) -> str | npt.NDArray[np.uint8] | None:
        # Instantiate pygame if not already done
        if self.window is None and self.render_mode == "human":
            window_w, window_h, _ = self._get_pygame_window_square_dimension()
            pygame.init()
            self.window = pygame.display.set_mode((window_w, window_h))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            assert self.window, f"No pygame window instantiated"
            assert self.clock, f"No pygame clock instantiated"

            # Copy the canvas to the window
            self.window.blit(
                self._render_frame_human_pygame_canvas_gen(), (0, 0)
            )
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array_mode()
        else:  # mode == "ansi"
            return self._render_frame_ansi_mode()

    def _render_frame_human_pygame_canvas_gen(self) -> pygame.Surface:
        # Create a canvas
        (
            window_w,
            window_h,
            square_size,
        ) = self._get_pygame_window_square_dimension()
        canvas = pygame.Surface((window_w, window_h))
        canvas.fill((255, 255, 255))

        # Colour the start cell
        pygame.draw.rect(
            canvas,
            START_CELL_COLOUR,
            (self._curr_start_state * square_size, 0, square_size, square_size),
        )

        # Colour the goal cell
        pygame.draw.rect(
            canvas,
            GOAL_CELL_COLOUR,
            (self.goal_state * square_size, 0, square_size, square_size),
        )

        # Colour the special cells
        for s in self.special_states:
            pygame.draw.rect(
                canvas,
                SPECIAL_CELL_COLOUR,
                (s * square_size, 0, square_size, square_size),
            )

        # Draw the agent
        agent_center_w = self._agent_location * square_size + square_size // 2
        agent_center_h = square_size // 2
        pygame.draw.circle(
            canvas,
            AGENT_COLOUR,
            (agent_center_w, agent_center_h),
            PYGAME_DEFAULT_AGENT_RADIUS,
        )

        # Add grid lines
        for i in range(self.corridor_length):
            pygame.draw.line(
                canvas,
                LINE_COLOUR,
                (i * square_size, 0),
                (i * square_size, i * square_size),
                2,
            )
        return canvas

    def _render_frame_ansi_mode(self) -> str:
        # Agent: *
        # Start state: S
        # Goal state: G
        # Special states: ^
        base_str = [" "] * self.corridor_length
        base_str[self._curr_start_state] = "S"
        base_str[self.goal_state] = "G"
        for s in self.special_states:
            base_str[s] = "^"
        base_str[self._agent_location] = "*"

        return "[" + "|".join(base_str) + "]"

    def _render_rgb_array_mode(self) -> npt.NDArray[np.uint8]:
        return pygame.surfarray.array3d(
            self._render_frame_human_pygame_canvas_gen()
        )

    def _get_distance_to_goal(self) -> int:
        return abs(self._agent_location - self.goal_state)

    def _get_action_str(
        self, action: ActionIntType | ActionNpIntType | ActionStrType
    ) -> ActionStrType:
        if isinstance(action, ActionStrType):
            assert action in self.ACTION, f"Invalid action: {action}"
            return action
        else:  # int types
            assert action in range(
                len(self.ACTION)
            ), f"Invalid action index: {action}"
            return self.ACTION[action]

    def _convert_action_to_movement(
        self, action: ActionIntType | ActionNpIntType | ActionStrType
    ) -> int:
        action = self._get_action_str(action)
        if self._agent_location in self.special_states:
            # In special state, left will move you to the right
            return {"L": 1, "R": -1}[action]
        else:
            # Normal case left will move you to the left
            return {"L": -1, "R": 1}[action]

    def _get_agent_new_location(
        self, action: ActionIntType | ActionNpIntType | ActionStrType
    ) -> int:
        new_p = self._agent_location + self._convert_action_to_movement(action)
        return max(0, min(new_p, self.corridor_length - 1))

    def _get_agent_observation(self) -> ObservationType:
        # Return the wall information: since there is no up and down action,
        # we just need to return the left and right wall
        wall_status = [0, 0]
        if self._agent_location == 0:
            wall_status[0] = 1
        elif self._agent_location == (self.corridor_length - 1):
            wall_status[1] = 1
        return {
            "wall_status": np.array(wall_status, dtype=np.int64),
            "agent_location": self._agent_location,
        }
