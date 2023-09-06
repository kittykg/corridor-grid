from typing import Any, Optional

from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv


class SmallCorridorEnv(BaseSpecialStateCorridorEnv):
    """
    The basic small corridor environment.
    This is a special case of the base `BaseSpecialStateCorridorEnv` with the following
    configuration:
    State 0 (start), State 1 (special state), State 2, State 3 (goal state)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        truncate_tolerance: Optional[int] = None,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            customisation_cfg_dict={
                "corridor_length": 4,
                "start_state": 0,
                "goal_state": 3,
                "special_states": [1],
            },
        )
        if truncate_tolerance:
            self.truncate_tolerance = truncate_tolerance
            self.customisation_cfg = (
                self.customisation_cfg.adjust_truncate_tolerance(
                    truncate_tolerance
                )
            )


class LongCorridorEnv(BaseSpecialStateCorridorEnv):
    """
    The fully customisable corridor environment.
    Can adjust the length, the start, the goal and the special states of the
    environment. To customise it, pass in a dict with the following keys:
    `corridor_length`, `start_state`, `goal_state` and `special_states`. If any
    of the keys are missing, the missing field would be default as
    `corridor_length = 4`, `start_state = 0`, `goal_state = 3` and
    `special_states = [1]`. If not customised (i.e. no dict is passed in), it
    will be default to `SmallCorridorEnv` (4 states, start at 0, goal at 3,
    special state at 1 only).
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        customisation_cfg_dict: dict[str, Any] = dict(),
    ) -> None:
        super().__init__(render_mode, customisation_cfg_dict)


# if __name__ == "__main__":
#     # env = SmallCorridorEnv(render_mode="human")
#     env = LongCorridorEnv(
#         render_mode="human",
#         customisation_cfg_dict={
#             "corridor_length": 5,
#             "start_state": 0,
#             "goal_state": 4,
#             "special_states": [2, 3],
#         },
#     )
#     env.reset()
#     input("Type anything in the console to quit.\n")
