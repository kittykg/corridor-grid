import unittest

import numpy as np

from corridor_grid.envs import SmallSSCorridorEnv, LongSSCorridorEnv


class TestSmallSSCorridorEnv(unittest.TestCase):
    def test_movement(self):
        env = SmallSSCorridorEnv(render_mode="ansi")
        env.reset()
        # At the beginning the agent is at state 0
        self.assertEqual(env._agent_location, 0)

        # From state 0, move left doesn't move the agent
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 0)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([1, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 3)

        # From state 0, move right moves the agent to state 1
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 1)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 2)

        # From state 1, move right moves the agent to state 0
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 0)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([1, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 3)

        # From state 1, move left moves the agent to state 2
        env.step(1)  # Move to state 1 first from state 0
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 2)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 1)

        # From state 2, move left moves the agent to state 1
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 1)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
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
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 1], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 0)

    def test_the_episode_will_be_truncated(self):
        # Default truncated tolerance is 50
        env = SmallSSCorridorEnv(render_mode="ansi")
        env.reset()
        for _ in range(49):
            obs, reward, done, truncated, _ = env.step(0)
            self.assertEqual(obs["agent_location"], 0)
            np.testing.assert_array_equal(
                obs["wall_status"], np.array([1, 0], dtype=np.int64)
            )
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(reward, -1)
        obs, reward, done, truncated, _ = env.step(0)
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(obs["agent_location"], 0)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([1, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)

        # We can create another environment with a different truncated tolerance
        env = SmallSSCorridorEnv(render_mode="ansi", truncate_tolerance=13)
        env.reset()
        for _ in range(12):
            obs, reward, done, truncated, _ = env.step(0)
            self.assertEqual(obs["agent_location"], 0)
            np.testing.assert_array_equal(
                obs["wall_status"], np.array([1, 0], dtype=np.int64)
            )
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(reward, -1)
        obs, reward, done, truncated, _ = env.step(0)
        self.assertEqual(obs["agent_location"], 0)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([1, 0], dtype=np.int64)
        )
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(reward, -1)


class TestLongSSCorridorEnv(unittest.TestCase):
    def test_bad_customisations_raise_assertion_error(self):
        # Corridor length should be be at least 2
        with self.assertRaises(
            AssertionError,
            msg="Corridor length must be at least 2 (a start and a goal)",
        ):
            env = LongSSCorridorEnv(
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
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": -1,
                },
            )

        # Start state should be in [0, corridor_length - 1]
        with self.assertRaises(
            AssertionError, msg="Start state must be non-negative"
        ):
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 3,
                    "start_state": -1,
                },
            )
        with self.assertRaises(
            AssertionError, msg="Start state must be within the corridor"
        ):
            env = LongSSCorridorEnv(
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
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 3,
                    "goal_state": -1,
                },
            )
        with self.assertRaises(
            AssertionError, msg="Goal state must be within the corridor"
        ):
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "corridor_length": 2,
                    "goal_state": 3,
                },
            )
        with self.assertRaises(
            AssertionError, msg="Goal state must be different from start state"
        ):
            env = LongSSCorridorEnv(
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
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "special_states": [2, -4],
                },
            )
        with self.assertRaises(
            AssertionError,
            msg="Special state must be within the corridor, get get 8",
        ):
            env = LongSSCorridorEnv(
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
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "truncate_tolerance": -1,
                },
            )
        with self.assertRaises(
            AssertionError,
            msg="Truncate tolerance must be positive",
        ):
            env = LongSSCorridorEnv(
                render_mode="ansi",
                customisation_cfg_dict={
                    "truncate_tolerance": 0,
                },
            )

    def test_no_configuration_dict_should_default_to_small_corridor(self):
        env1 = LongSSCorridorEnv(render_mode="ansi")
        env2 = SmallSSCorridorEnv(render_mode="ansi")

        self.assertEqual(env1.corridor_length, env2.corridor_length)
        # Start state is random (stored as `None`) for LongSSCorridorEnv when no
        # configuration dict is provided
        self.assertIsNone(env1.start_state)
        self.assertEqual(env1.goal_state, env2.goal_state)
        self.assertEqual(env1.special_states, env2.special_states)
        self.assertEqual(env1.truncate_tolerance, env2.truncate_tolerance)

    def test_configuration_dict_with_missing_fields_will_fall_to_default(self):
        env = LongSSCorridorEnv(
            render_mode="ansi",
            customisation_cfg_dict={
                "corridor_length": 6,
                "start_state": 0,
            },
        )
        self.assertEqual(env.corridor_length, 6)
        self.assertEqual(env.start_state, 0)
        self.assertEqual(env.goal_state, 5)
        self.assertEqual(env.special_states, [1])

    def test_long_corridor_ansi_representation(self):
        env = LongSSCorridorEnv(
            render_mode="ansi",
            customisation_cfg_dict={
                "corridor_length": 6,
                "start_state": 4,
                "goal_state": 1,
                "special_states": [2, 4],
            },
        )
        env.reset()

        # At the beginning, the agent is at state 4
        self.assertEqual(env._agent_location, 4)
        self.assertEqual(env.render(), "[ |G|^| |*| ]")

        # From state 4, move left move the agent to state 5
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 5)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 1], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 4)
        self.assertEqual(env.render(), "[ |G|^| |^|*]")

        # From state 5, move left move the agent to state 4
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 4)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 3)
        self.assertEqual(env.render(), "[ |G|^| |*| ]")

        # From state 4, move right move the agent to state 3
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 3)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 2)
        self.assertEqual(env.render(), "[ |G|^|*|^| ]")

        # From state 3, move left move the agent to state 2
        obs, reward, done, truncated, info = env.step(0)
        self.assertEqual(obs["agent_location"], 2)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "L")
        self.assertEqual(info["distance_to_goal"], 1)
        self.assertEqual(env.render(), "[ |G|*| |^| ]")

        # From state 2, move right move the agent to state 1
        obs, reward, done, truncated, info = env.step(1)
        self.assertEqual(obs["agent_location"], 1)
        np.testing.assert_array_equal(
            obs["wall_status"], np.array([0, 0], dtype=np.int64)
        )
        self.assertEqual(reward, -1)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "R")
        self.assertEqual(info["distance_to_goal"], 0)
        self.assertEqual(env.render(), "[ |*|^| |^| ]")


if __name__ == "__main__":
    unittest.main()
