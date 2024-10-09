from gymnasium.envs.registration import register

__version__ = "0.0.1"


# Small Corridor (SC)
# Start at state 0, special state at state1, goal at state 3
# ----------------------------------------

register(
    id="CG-SC-v0",
    entry_point="corridor_grid.envs:SmallSSCorridorEnv",
)

# Long Corridor 5 (LC5)
# Corridor length 5
# Start at state 0
# Special state at state 1
# Goal at state 4
# ----------------------------------------
register(
    id="CG-LC5-v0",
    entry_point="corridor_grid.envs:LongSSCorridorEnv",
    kwargs={"customisation_cfg_dict": {"corridor_length": 5, "start_state": 0}},
)

# Long Corridor 5 Special State 2 (LC5-S2)
# Corridor length 5
# Start at state 0
# Special state at state 2
# Goal at state 4
# ----------------------------------------
register(
    id="CG-LC5-S2-v0",
    entry_point="corridor_grid.envs:LongSSCorridorEnv",
    kwargs={
        "customisation_cfg_dict": {
            "corridor_length": 5,
            "start_state": 0,
            "special_states": [2],
        }
    },
)

# Long Corridor 11 (LC11)
# Corridor length 11
# Start at state 7
# Special state at state 5, 6, 7, 8
# Goal at state 3
# ----------------------------------------
register(
    id="CG-LC11-v0",
    entry_point="corridor_grid.envs:LongSSCorridorEnv",
    kwargs={
        "customisation_cfg_dict": {
            "corridor_length": 11,
            "start_state": 7,
            "goal_state": 3,
            "special_states": [5, 6, 7, 8],
        }
    },
)

# Circular Corridor 11 (CC11)
# Corridor length 11
# Start at state 9
# Special state at state 1, 2, 10
# Goal at state 3
register(
    id="CG-CC11-v0",
    entry_point="corridor_grid.envs:CircularSSCorridorEnv",
    kwargs={
        "customisation_cfg_dict": {
            "corridor_length": 11,
            "start_state": 9,
            "goal_state": 3,
            "special_states": [1, 2, 10],
        }
    },
)

# Door Corridor 5 (DC5)
# Corridor length 5
# Start at state 0
# Goal at state 4
# Agent facing up at the beginning
# Doors at state 1, 2, 3
register(
    id="CG-DC5-v0",
    entry_point="corridor_grid.envs:DoorCorridorEnv",
)

# Door Corridor T 5 (DCT5)
# Corridor length 5
# Start at state 0
# Goal at state 4
# Agent facing up at the beginning
# Doors at state 1, 2, 3
register(
    id="CG-DCT5-v0",
    entry_point="corridor_grid.envs:DoorCorridorTEnv",
)

# Door Corridor OT 5 (DCOT5)
# Corridor length 5
# Start at state 0
# Goal at state 4
# Agent facing up at the beginning
# Doors at state 1, 2, 3
register(
    id="CG-DCOT5-v0",
    entry_point="corridor_grid.envs:DoorCorridorOTEnv",
)
