import math
from typing import Any

import numpy as np
import pygame


from corridor_grid.envs.base_ss_corridor import (
    ActionIntType,
    ActionNpIntType,
    ActionStrType,
    ObservationType,
    BaseSpecialStateCorridorEnv,
)

PYGAME_WINDOW_SIZE = 500
BIG_CIRCLE_RADIUS = 250
BIG_CIRCLE_COLOUR = (156, 194, 255, 160)
SMALL_WHITE_CIRCLE_RADIUS = 150
AGENT_LOCATION_CIRCLE_RADIUS = 180
AGENT_RADIUS = 10
TEXT_LOCATION_CIRCLE_RADIUS = 230
FONT_SIZE = 24

LINE_COLOUR = (0, 0, 0, 160)
START_STATE_COLOUR = (255, 128, 0, 160)
GOAL_STATE_COLOUR = (0, 128, 255, 160)
SPECIAL_STATE_COLOUR = (128, 0, 255, 160)
AGENT_COLOUR = (0, 255, 0, 255)
TEXT_COLOUR = (0, 0, 0, 160)


class CircularSSCorridorEnv(BaseSpecialStateCorridorEnv):
    """
    A circular corridor environment where the states are connected in a circle.
    The corridor does not have an 'end'.
    The states are arranged in clockwise order. Moving left is equivalent of
    moving one step anti-clockwise, and moving right is equivalent of moving one
    step clockwise.
    Special states still reverse the actual movement: moving right (clockwise)
    will actually move left (anti-clockwise) and vice versa.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        customisation_cfg_dict: dict[str, Any] = dict(),
    ) -> None:
        super().__init__(render_mode, customisation_cfg_dict)

    # Overridden functions
    def _get_pygame_window_square_dimension(self) -> tuple[int, int, int]:
        # Since the corridor is circular, we don't have square size any more.
        # The last return value is now a placeholder value and should not be
        # used.
        return PYGAME_WINDOW_SIZE, PYGAME_WINDOW_SIZE, PYGAME_WINDOW_SIZE

    def _render_frame_human_pygame_canvas_gen(self) -> pygame.Surface:
        canvas = pygame.Surface((PYGAME_WINDOW_SIZE, PYGAME_WINDOW_SIZE))
        canvas.fill((255, 255, 255))

        font = pygame.font.Font("freesansbold.ttf", FONT_SIZE)
        centre = (PYGAME_WINDOW_SIZE // 2, PYGAME_WINDOW_SIZE // 2)

        def calculate_end_point(
            angle: float,
            center: tuple[int, int] = centre,
            radius: int = BIG_CIRCLE_RADIUS,
        ) -> tuple[int, int]:
            x = center[0] + radius * math.cos(math.radians(angle))
            y = center[1] + radius * math.sin(math.radians(angle))
            return (int(x), int(y))

        # Draw the big (background) circle
        pygame.draw.circle(canvas, BIG_CIRCLE_COLOUR, centre, BIG_CIRCLE_RADIUS)

        pie_slice_degrees = 360 / self.corridor_length

        # Draw the pie slices
        for i in range(self.corridor_length):
            angle = i * pie_slice_degrees
            # Draw the separation line
            pygame.draw.line(
                canvas,
                LINE_COLOUR,
                centre,
                calculate_end_point(angle),
                2,
            )

            # Draw the state number
            text_color = TEXT_COLOUR
            if i in self.special_states:
                text_color = SPECIAL_STATE_COLOUR
            elif i == self._curr_start_state:
                text_color = START_STATE_COLOUR
            elif i == self.goal_state:
                text_color = GOAL_STATE_COLOUR

            text_str = f"S{i}"
            if i in self.special_states:
                text_str = f"*S{i}"
            text = font.render(text_str, True, text_color, None)
            text_rect = text.get_rect(
                center=calculate_end_point(
                    (i + 0.5) * pie_slice_degrees,
                    radius=TEXT_LOCATION_CIRCLE_RADIUS,
                )
            )
            canvas.blit(text, text_rect)

        # Draw the agent
        pygame.draw.circle(
            canvas,
            AGENT_COLOUR,
            calculate_end_point(
                (self._agent_location + 0.5) * pie_slice_degrees,
                radius=AGENT_LOCATION_CIRCLE_RADIUS,
            ),
            AGENT_RADIUS,
        )

        # Draw the small white circle to cover the middle of the big circle, so
        # that it looks like a ring
        pygame.draw.circle(
            canvas, (255, 255, 255), centre, SMALL_WHITE_CIRCLE_RADIUS
        )

        return canvas

    def _get_distance_to_goal(self) -> int:
        # Each state can have two distance to the goal state, depending counting
        # clockwise or counter-clockwise. We take the minimum of the two.
        diff = abs(self._agent_location - self.goal_state)
        return min(diff, self.corridor_length - diff)

    def _get_agent_new_location(
        self, action: ActionIntType | ActionNpIntType | ActionStrType
    ) -> int:
        new_p = self._agent_location + self._convert_action_to_movement(action)
        return new_p % self.corridor_length

    def _get_agent_observation(self) -> ObservationType:
        # Return the wall information: since there is no up and down action,
        # we just need to return the left and right wall. And since the
        # circular corridor does not have an end, both left and right wall will
        # always be 0.
        wall_status = [0, 0]
        return {
            "wall_status": np.array(wall_status, dtype=np.int64),
            "agent_location": self._agent_location,
        }
