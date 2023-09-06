from gymnasium.envs.registration import register

from corridor_grid.envs.ss_corridor import SmallCorridorEnv, LongCorridorEnv

__version__ = "0.0.1"


def register_minigrid_envs():
    # Small Corridor (SC)
    # Start at state 0, special state at state1, goal at state 3
    # ----------------------------------------

    register(
        id="CG-SC-v0",
        entry_point="corridor_grid.envs:SmallCorridorEnv",
    )

    # Long Corridor 5 (LC5)
    # Corridor length 5
    # Start at state 0
    # Special state at state 1
    # Goal at state 4
    # ----------------------------------------
    register(
        id="CG-LC5-v0",
        entry_point="minigrid.envs:LongCorridorEnv",
        kwargs={
            "customisation_cfg_dict": {"corridor_length": 5, "start_state": 0}
        },
    )

    # Long Corridor 5 Special State 2 (LC5-S2)
    # Corridor length 5
    # Start at state 0
    # Special state at state 2
    # Goal at state 4
    # ----------------------------------------
    register(
        id="CG-LC5-S2-v0",
        entry_point="minigrid.envs:LongCorridorEnv",
        kwargs={
            "customisation_cfg_dict": {
                "corridor_length": 5,
                "start_state": 0,
                "special_states": [2],
            }
        },
    )

    # Long Corridor 11 (LC11)
    # Start at state 7
    # Special state at state 5, 6, 7, 8
    # Goal at state 3
    # ----------------------------------------
    register(
        id="CG-LC11-v0",
        entry_point="minigrid.envs:LongCorridorEnv",
        kwargs={
            "customisation_cfg_dict": {
                "corridor_length": 11,
                "start_state": 7,
                "goal_state": 3,
                "special_states": [5, 6, 7, 8],
            }
        },
    )
