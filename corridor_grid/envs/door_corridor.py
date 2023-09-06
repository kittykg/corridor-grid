from enum import IntEnum
import math
from typing import Any, Callable

import gymnasium as gym
from minigrid.utils.rendering import (
    highlight_img,
    point_in_rect,
    point_in_circle,
    point_in_triangle,
    rotate_fn,
)
import numpy as np
import numpy.typing as npt
import pygame

ActionIntType = int
ObservationType = dict[str, Any]

DEFAULT_TRUNCATE_TOLERANCE = 270
DEFAULT_AGENT_VIEW_SIZE = 3


class DoorCorridorAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    TOGGLE = 3


class AgentDirection(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def turn_left(self) -> "AgentDirection":
        return AgentDirection((self - 1) % 4)

    def turn_right(self) -> "AgentDirection":
        return AgentDirection((self + 1) % 4)


class Object(IntEnum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    DOOR = 3
    AGENT = 4
    GOAL = 5


class State(IntEnum):
    OPEN = 0
    CLOSED = 1

    def toggle(self) -> "State":
        return State((self + 1) % 2)


class DoorCorridorEnv(gym.Env[dict[str, Any], int]):
    """
    A simplified environment similar to the KeyCorridorS3R1-v0 environment.
    There are 4 available actions:
    0 - left - Turn left
    1 - right - Turn right
    2 - forward - Move forward
    3 - toggle - Toggle/activate an object
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    render_mode: str | None
    corridor_length: int
    agent_view_size: int
    max_steps: int

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict

    grid_width: int
    grid_height: int
    start_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    grid: npt.NDArray[np.uint8]

    agent_pos: tuple[int, int]
    agent_dir: AgentDirection
    _step_count: int

    window: pygame.Surface | None
    clock: pygame.time.Clock | None

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = DEFAULT_TRUNCATE_TOLERANCE,
        corridor_length: int = 5,
        agent_view_size: int = DEFAULT_AGENT_VIEW_SIZE,
    ) -> None:
        if render_mode:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}. "
        self.render_mode = render_mode
        self.corridor_length = corridor_length

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(len(DoorCorridorAction))
        image_obs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.agent_view_size,
                self.agent_view_size,
                2,
            ),
            # the last channel of `shape` is of size 2 (object + state), as we
            # do not have colour in this environment
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "image": image_obs_space,
                "direction": gym.spaces.Discrete(len(AgentDirection)),
            }
        )

        self.grid_width = corridor_length + 2
        self.grid_height = 3

        self.start_pos = (1, 1)
        self.goal_pos = (self.grid_width - 2, 1)

        self._gen_grid(width=self.grid_width, height=self.grid_height)

        self.agent_pos = self.start_pos
        self.agent_dir = AgentDirection.UP
        self._step_count = 0

        self.window = None
        self.clock = None

    def step(
        self, action: ActionIntType
    ) -> tuple[ObservationType, float, bool, bool, dict[str, Any]]:
        assert (
            action in DoorCorridorAction._value2member_map_
        ), f"Invalid action {action}"

        self._step_count += 1

        if action == DoorCorridorAction.LEFT:
            self.agent_dir = self.agent_dir.turn_left()
        elif action == DoorCorridorAction.RIGHT:
            self.agent_dir = self.agent_dir.turn_right()
        elif action == DoorCorridorAction.FORWARD:
            i_x, i_y = self._in_front_of_agent_coord()
            # Only when the agent is facing an open state that is not a wall can
            # the agent actually move forward
            if (
                self._with_grid(i_x, i_y)
                and self.grid[i_y, i_x, 1] == State.OPEN
                and self.grid[i_y, i_x, 0] != Object.WALL
            ):
                self.agent_pos = (i_x, i_y)
        else:
            # Toggle
            i_x, i_y = self._in_front_of_agent_coord()
            # Only when the agent is facing a door can the agent toggle it
            if (
                self._with_grid(i_x, i_y)
                and self.grid[i_y, i_x, 0] == Object.DOOR
            ):
                self.grid[i_y, i_x, 1] = State(self.grid[i_y, i_x, 1]).toggle()

        agent_pov = self._get_agent_pov()
        terminated = self.agent_pos == self.goal_pos
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return (
            {
                "image": agent_pov,
                "direction": self.agent_dir.value,
            },
            -1,
            terminated,
            truncated,
            {
                "action": DoorCorridorAction(action).name,
                "agent_direction": self.agent_dir.name,
            },
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationType, dict[str, Any]]:
        self.agent_pos = self.start_pos
        self.agent_dir = AgentDirection.UP
        self._step_count = 0
        self._gen_grid(width=self.grid_width, height=self.grid_height)

        agent_pov = self._get_agent_pov()

        if self.render_mode == "human":
            self.render()

        return {
            "image": agent_pov,
            "direction": self.agent_dir.value,
        }, {}

    def render(self) -> None | npt.NDArray[np.uint8]:
        full_render_img = self.get_full_render(tile_size=30)
        if self.render_mode == "human":
            self._render_human_mode(full_render_img)
        else:
            return full_render_img

    def get_full_render(self, tile_size: int = 10) -> npt.NDArray[np.uint8]:
        highlight_mask = np.zeros(
            (self.grid_height * tile_size, self.grid_width * tile_size),
            dtype=bool,
        )
        top_x, top_y = self._get_view_exts()
        for j in range(self.agent_view_size):
            for i in range(self.agent_view_size):
                x = top_x + i
                y = top_y + j

                if self._with_grid(x, y):
                    highlight_mask[y, x] = True

        img = np.zeros(
            (self.grid_height * tile_size, self.grid_width * tile_size, 3),
            dtype=np.uint8,
        )
        for j in range(self.grid_height):
            for i in range(self.grid_width):
                agent_here = (i, j) == self.agent_pos
                tile_img = render_tile(
                    object=Object(self.grid[j, i, 0]),
                    state=State(self.grid[j, i, 1]),
                    agent_dir=self.agent_dir if agent_here else None,
                    highlight=highlight_mask[j, i],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def get_pov_render(self, tile_size: int = 10) -> npt.NDArray[np.uint8]:
        agent_pov = self._get_agent_pov()
        img_w = self.agent_view_size * tile_size
        img = np.zeros((img_w, img_w, 3), dtype=np.uint8)

        pov_agent_pos = (self.agent_view_size // 2, self.agent_view_size - 1)

        for j in range(self.agent_view_size):
            for i in range(self.agent_view_size):
                agent_here = (i, j) == pov_agent_pos
                tile_img = render_tile(
                    object=Object(agent_pov[j, i, 0]),
                    state=State(agent_pov[j, i, 1]),
                    agent_dir=AgentDirection.UP if agent_here else None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            pygame.quit()

    def _in_front_of_agent_coord(self) -> tuple[int, int]:
        if self.agent_dir == AgentDirection.RIGHT:
            return (self.agent_pos[0] + 1, self.agent_pos[1])
        elif self.agent_dir == AgentDirection.DOWN:
            return (self.agent_pos[0], self.agent_pos[1] + 1)
        elif self.agent_dir == AgentDirection.LEFT:
            return (self.agent_pos[0] - 1, self.agent_pos[1])
        elif self.agent_dir == AgentDirection.UP:
            return (self.agent_pos[0], self.agent_pos[1] - 1)
        else:
            assert False, "invalid agent direction"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = np.zeros((height, width, 2), dtype=np.uint8)

        self.grid[:, :, 0] = Object.EMPTY

        self.grid[:, 0, 0] = Object.WALL
        self.grid[:, -1, 0] = Object.WALL
        self.grid[0, :, 0] = Object.WALL
        self.grid[-1, :, 0] = Object.WALL

        # The doors
        self.grid[1, 2:-2] = [Object.DOOR, State.CLOSED]

        # The agent
        self.grid[self.start_pos[1], self.start_pos[0], 0] = Object.AGENT

        # The goal
        self.grid[self.goal_pos[1], self.goal_pos[0], 0] = Object.GOAL

    def _get_agent_pov(self) -> npt.NDArray[np.uint8]:
        agent_pov = np.zeros(
            (self.agent_view_size, self.agent_view_size, 2), dtype=np.uint8
        )

        top_x, top_y = self._get_view_exts()
        for j in range(self.agent_view_size):
            for i in range(self.agent_view_size):
                x = top_x + i
                y = top_y + j

                if self._with_grid(x, y):
                    agent_pov[j, i] = self.grid[y, x]
                else:
                    agent_pov[j, i, 0] = Object.UNSEEN

        # Rotate if needed
        if self.agent_dir != AgentDirection.UP:
            agent_pov = np.rot90(agent_pov, k=self.agent_dir + 1)

        # A closed door in front of the agent would block the agent's view
        agent_x_in_pov = self.agent_view_size // 2
        for y in range(self.agent_view_size - 2, 0, -1):
            if np.all(
                agent_pov[y, agent_x_in_pov] == [Object.DOOR, State.CLOSED]
            ):
                agent_pov[:y, :] = Object.UNSEEN
                break

        return agent_pov

    def _with_grid(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def _get_view_exts(self) -> tuple[int, int]:
        # Facing right
        if self.agent_dir == AgentDirection.RIGHT:
            top_x = self.agent_pos[0]
            top_y = self.agent_pos[1] - self.agent_view_size // 2
        # Facing down
        elif self.agent_dir == AgentDirection.DOWN:
            top_x = self.agent_pos[0] - self.agent_view_size // 2
            top_y = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == AgentDirection.LEFT:
            top_x = self.agent_pos[0] - self.agent_view_size + 1
            top_y = self.agent_pos[1] - self.agent_view_size // 2
        # Facing up
        elif self.agent_dir == AgentDirection.UP:
            top_x = self.agent_pos[0] - self.agent_view_size // 2
            top_y = self.agent_pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        return top_x, top_y

    def _render_human_mode(
        self, full_render_img: npt.NDArray[np.uint8]
    ) -> None:
        if self.window is None and self.render_mode == "human":
            window_w = full_render_img.shape[1]
            window_h = full_render_img.shape[0]
            pygame.init()
            self.window = pygame.display.set_mode((window_w, window_h))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        assert self.window, f"No pygame window instantiated"
        assert self.clock, f"No pygame clock instantiated"

        surf = pygame.surfarray.make_surface(full_render_img.swapaxes(0, 1))
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])


# The functions below are adapted from the rendering functions in the minigrid
# library.
COLOURS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}


def render_tile(
    object: Object,
    state: State,
    agent_dir: AgentDirection | None = None,
    highlight: bool = False,
    tile_size: int = 10,
) -> npt.NDArray[np.uint8]:
    img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

    if object == Object.WALL:
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLOURS["grey"])
    elif object == Object.DOOR:
        c = COLOURS["green"]

        if state == State.OPEN:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
    elif object == Object.GOAL:
        fill_coords(img, point_in_rect(0, 1, 0, 1), 0.5 * COLOURS["green"])

    # Overlay the agent on top
    if agent_dir is not None:
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir
        )
        fill_coords(img, tri_fn, (255, 0, 0))

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    # Highlight the cell if needed
    if highlight:
        highlight_img(img)

    return img


def fill_coords(
    img: npt.NDArray[np.uint8],
    fn: Callable[[float, float], bool],
    colour: npt.NDArray | tuple[int, int, int],
) -> npt.NDArray[np.uint8]:
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = colour

    return img
