import unittest

import numpy as np
from corridor_grid.envs.door_corridor import (
    DoorCorridorEnv,
    AgentDirection,
    Object,
    State,
)


class TestDoorCorridorEnv(unittest.TestCase):
    def test_initial_grid(self):
        env = DoorCorridorEnv()
        env.reset()
        self.assertEqual(env.action_space.n, 4)
        np.testing.assert_equal(
            env.grid,
            np.array(
                [
                    [
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.AGENT, State.OPEN],
                        [Object.DOOR, State.CLOSED],
                        [Object.DOOR, State.CLOSED],
                        [Object.DOOR, State.CLOSED],
                        [Object.GOAL, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                ],
                dtype=np.uint8,
            ),
        )

    def test_get_agent_poz(self):
        env = DoorCorridorEnv()
        env.reset()
        # Agent facing up
        np.testing.assert_equal(
            env._get_agent_pov(),
            np.array(
                [
                    [
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.AGENT, State.OPEN],
                        [Object.DOOR, State.CLOSED],
                    ],
                ],
                dtype=np.uint8,
            ),
        )

        # Agent facing right
        env.agent_dir = AgentDirection.RIGHT
        np.testing.assert_equal(
            env._get_agent_pov(),
            np.array(
                [
                    [
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.DOOR, State.CLOSED],
                        [Object.WALL, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.AGENT, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                ],
                dtype=np.uint8,
            ),
        )

        # Agent facing down
        env.agent_dir = AgentDirection.DOWN
        np.testing.assert_equal(
            env._get_agent_pov(),
            np.array(
                [
                    [
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                    [
                        [Object.DOOR, State.CLOSED],
                        [Object.AGENT, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                ],
                dtype=np.uint8,
            ),
        )

        # Agent facing left
        env.agent_dir = AgentDirection.LEFT
        np.testing.assert_equal(
            env._get_agent_pov(),
            np.array(
                [
                    [
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                        [Object.UNSEEN, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                    [
                        [Object.WALL, State.OPEN],
                        [Object.AGENT, State.OPEN],
                        [Object.WALL, State.OPEN],
                    ],
                ],
                dtype=np.uint8,
            ),
        )
