import unittest

import numpy as np

from corridor_grid.envs import SmallSSCorridorEnv
from corridor_grid.envs import CircularSSCorridorEnv


class TestCircularSSCorridorEnv(unittest.TestCase):
    wall_status = np.array([0, 0], dtype=np.int64)

    def test_bad_customisations_raise_assertion_error(self):
        # Corridor length should be be at least 2
        with self.assertRaises(
            AssertionError,
            msg="Corridor length must be at least 2 (a start and a goal)",
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 1,
                    "start_state": 0,
                    "goal_state": 0,
                },
            )
        with self.assertRaises(
            AssertionError,
            msg="Corridor length must be at least 2 (a start and a goal)",
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": -1,
                },
            )

        # Start state should be in [0, corridor_length - 1]
        with self.assertRaises(
            AssertionError, msg="Start state must be non-negative"
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 3,
                    "start_state": -1,
                },
            )
        with self.assertRaises(
            AssertionError, msg="Start state must be within the corridor"
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 6,
                    "start_state": 9,
                },
            )

        # Goal state should be in [0, corridor_length - 1], and should be
        # different from the start state
        with self.assertRaises(
            AssertionError, msg="Goal state must be non-negative"
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 3,
                    "goal_state": -1,
                },
            )
        with self.assertRaises(
            AssertionError, msg="Goal state must be within the corridor"
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 2,
                    "goal_state": 3,
                },
            )
        with self.assertRaises(
            AssertionError, msg="Goal state must be different from start state"
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 2,
                    "start_state": 0,
                    "goal_state": 0,
                },
            )

        # Each special state should be in [0, corridor_length - 1]
        with self.assertRaises(
            AssertionError, msg="Special state must be non-negative but get -4"
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "special_states": [2, -4],
                },
            )
        with self.assertRaises(
            AssertionError,
            msg="Special state must be within the corridor, get get 8",
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "special_states": [1, 8],
                },
            )

        # Truncate tolerance should be positive
        with self.assertRaises(
            AssertionError,
            msg="Truncate tolerance must be positive",
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "truncate_tolerance": -1,
                },
            )
        with self.assertRaises(
            AssertionError,
            msg="Truncate tolerance must be positive",
        ):
            env = CircularSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "truncate_tolerance": 0,
                },
            )

    def test_no_configuration_dict_should_default_to_small_corridor_but_circular(
        self,
    ):
        env1 = CircularSSCorridorEnv(render_mode="ansi")
        env2 = SmallSSCorridorEnv(render_mode="ansi")

        self.assertEqual(env1.corridor_length, env2.corridor_length)
        self.assertEqual(env1.goal_state, env2.goal_state)
        self.assertEqual(env1.special_states, env2.special_states)
        self.assertEqual(env1.truncate_tolerance, env2.truncate_tolerance)

    def test_configuration_dict_with_missing_fields_will_fall_to_default(self):
        env = CircularSSCorridorEnv(
            render_mode="ansi",
            customisation_cfg_dict={
                "corridor_length": 6,
            },
        )
        self.assertEqual(env.corridor_length, 6)
        self.assertEqual(env.goal_state, 5)
        self.assertEqual(env.special_states, [1])
        self.assertIsNone(env.start_state) # no start state specified

    def test_movement_small_circular_env_1(self):
        env = CircularSSCorridorEnv(
            render_mode="ansi", customisation_cfg_dict={"start_state": 0}
        )
        env.reset()
        # At the beginning the agent is at state 0
        self.assertEqual(env._agent_location, 0)

        # From state 0, moving left move agent to state 3, reaching the goal
        # `done` should be True
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 3)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 0)

    def test_movement_small_circular_env_2(self):
        env = CircularSSCorridorEnv(
            render_mode="ansi", customisation_cfg_dict={"start_state": 0}
        )
        env.reset()
        # At the beginning the agent is at state 0
        self.assertEqual(env._agent_location, 0)

        # From state 0, move right moves the agent to state 1
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 1)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 2)

        # From state 1, move right moves the agent to state 0
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 0)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 1)

        # From state 1, move left moves the agent to state 2
        env.step(1)  # Move to state 1 first from state 0
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 2)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 1)

        # From state 2, move left moves the agent to state 1
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 1)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 2)

        # From state 2, move right moves the agent to state 3, and `done` should
        # be true now
        env.step(0)  # Move to state 2 first from state 1
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 3)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 0)

    def test_movement_large_16_circular_env_1(self):
        env = CircularSSCorridorEnv(
            render_mode="ansi",
            customisation_cfg_dict={
                "corridor_length": 16,
                "start_state": 3,
                "goal_state": 9,
                "special_states": [5, 7],
            },
        )
        env.reset()
        # At the beginning the agent is at state 3
        self.assertEqual(env._agent_location, 3)

        # The ideal movement is: RRLRLR
        actions = [1, 1, 0, 1, 0, 1]
        action_str = ["R" if i == 1 else "L" for i in actions]
        states = [i for i in range(4, 10)]
        distance = [9 - s for s in states]

        for i in range(len(actions)):
            obs, reward, done, truncated, info = env.step(actions[i])
            np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
            self.assertEqual(obs["agent_location"], states[i])
            self.assertEqual(reward, -1)
            self.assertEqual(done, i == len(actions) - 1)
            self.assertFalse(truncated)
            self.assertEqual(info["action"], action_str[i])
            self.assertEqual(info["distance_to_goal"], distance[i])

    def test_movement_large_16_circular_env_2(self):
        env = CircularSSCorridorEnv(
            render_mode="ansi",
            customisation_cfg_dict={
                "corridor_length": 16,
                "start_state": 3,
                "goal_state": 9,
                "special_states": [5, 7],
            },
        )
        env.reset()
        # At the beginning the agent is at state 3
        self.assertEqual(env._agent_location, 3)

        # The other movement is LLLLLLLLLL
        actions = [0] * 10
        action_str = ["L"] * 10
        states = [(3 - i) % 16 for i in range(1, 11)]
        distance = [7, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        for i in range(len(actions)):
            obs, reward, done, truncated, info = env.step(actions[i])
            self.assertEqual(obs["agent_location"], states[i])
            np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
            self.assertEqual(reward, -1)
            self.assertEqual(done, i == len(actions) - 1)
            self.assertFalse(truncated)
            self.assertEqual(info["action"], action_str[i])
            self.assertEqual(info["distance_to_goal"], distance[i])

    def test_the_episode_will_be_truncated(self):
        # Default truncated tolerance is 50
        env = CircularSSCorridorEnv(render_mode="ansi", customisation_cfg_dict={"start_state": 0})
        env.reset()
        self.assertEqual(env._agent_location, 0)

        for i in range(49):
            obs, reward, done, truncated, info = env.step(1)
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(obs["agent_location"], int(i % 2 == 0))
            np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
            self.assertEqual(reward, -1)
            self.assertEqual(info["action"], "R")
            self.assertEqual(info["distance_to_goal"], 2 - i % 2)

        # After 49 moving right, the agent is at state 1
        obs, reward, done, truncated, _ = env.step(0)
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(obs["agent_location"], 2)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)

        # We can create another environment with a different truncated tolerance
        env = CircularSSCorridorEnv(
            render_mode="ansi",
            customisation_cfg_dict={"truncate_tolerance": 13, "start_state": 0},
        )
        env.reset()
        self.assertEqual(env._agent_location, 0)
        self.assertEqual(env.truncate_tolerance, 13)

        for i in range(12):
            obs, reward, done, truncated, info = env.step(1)
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(obs["agent_location"], int(i % 2 == 0))
            np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
            self.assertEqual(reward, -1)
            self.assertEqual(info["action"], "R")
            self.assertEqual(info["distance_to_goal"], 2 - i % 2)

        # After 13 moving right, we are at state 0
        obs, reward, done, truncated, _ = env.step(1)
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(obs["agent_location"], 1)
        np.testing.assert_array_equal(obs["wall_status"], self.wall_status)
        self.assertEqual(reward, -1)


if __name__ == "__main__":
    unittest.main()
