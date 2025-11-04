from __future__ import annotations
import numpy as np
from typing import Optional
from environments.grid_world import GridworldEnv


# Initialization
def initialize_state_values(env: GridworldEnv):
    r = env.height
    c = env.width
    Vs = np.zeros((r, c), dtype=np.float32)
    return Vs


# deterministic policy
def initialize_policy(env: GridworldEnv, use_good_policy=False):

    good_policy = {
        0: None,
        1: 3,
        2: 3,
        3: 2,
        4: 0,
        5: 0,
        6: 0,
        7: 2,
        8: 0,
        9: 0,
        10: 1,
        11: 2,
        12: 0,
        13: 1,
        14: 1,
        15: None,
    }
    if use_good_policy:
        return good_policy

    policy = {}
    terminal_states = [env.state_to_index(goal) for goal in env.goals]
    state_indices = [
        env.state_to_index((r, c)) for r in range(env.height) for c in range(env.width)
    ]

    n_actions = len(env.ACTIONS)

    for state_index in state_indices:
        if state_index in terminal_states:
            policy[state_index] = None
        else:
            policy[state_index] = np.random.choice(n_actions)
    return policy


# -------------------------Monte Carlo Helpers-------------------------#
class State_Data:
    def __init__(self, state_idx, action, reward, prev_state_obj: Optional[State_Data]):
        self.state_idx = state_idx
        self.action = action
        self.reward = reward
        self.prev_state_obj = prev_state_obj

        if self.prev_state_obj is None:
            self.step = 0
        else:
            self.step = self.prev_state_obj.step + 1


def generate_episode(
    env: GridworldEnv,
    policy: dict,
    start_state_idx: Optional[int] = None,
    start_action_idx: Optional[int]=None,
    is_policy_stochastic: bool = False,
    max_steps=100,
) -> State_Data:
    """
    Returns an object which contains the history of previous states
    """
    # pick a random state (terminal states excluded)
    state_indices = [
        idx
        for idx in range(env.width * env.height)
        if env.index_to_state(idx) not in env.goals
    ]

    # agent start index
    start_idx = (
        np.random.choice(state_indices) if start_state_idx is None else start_state_idx
    )

    # set agent initial position
    env.pos = env.index_to_state(start_idx)

    # follow the policy and move till the agent gets to a terminal state
    state_data = None
    next_idx = start_idx
    for i in range(max_steps):
        if i == 0 and start_action_idx is not None:
            action = start_action_idx
        else:
            if is_policy_stochastic:
                actions = list(env.ACTIONS.keys())
                action = np.random.choice(actions, p=policy[next_idx])
            else:
                # deterministic policy.
                action = policy[next_idx]

        new_state, reward, done, _ = env.step(action)

        state_data = State_Data(next_idx, action, reward, state_data)

        next_idx = env.state_to_index(new_state)

        if done:  # episode has ended.
            break
    if state_data is None:
        raise Exception("Failed in generating episode; state data should not be none")
    return state_data

def generate_state_action_keys(state_index, action_key):
    return (state_index, action_key)

def initialize_returns(env:GridworldEnv)->dict[int, list]:
    returns = {}
    actions = env.ACTIONS

    for state_index in range(env.width * env.height):
        for action_key in actions:
            key = generate_state_action_keys(state_index, action_key)
            returns[key] = []
    return returns

def initialize_action_value(env:GridworldEnv):
    Q_s_a = {}
    actions = env.ACTIONS
    state_indices = [idx for idx in range(env.width*env.height) if env.index_to_state(idx) not in env.goals ]

    for state_index in state_indices:
        for action_key in actions:
            key = generate_state_action_keys(state_index, action_key)
            Q_s_a[key] = np.random.rand() # generate a random value from [0, 1)
    return Q_s_a

def is_state_action_visited_previously(state_index, action_key, episode:State_Data):
  """moves down the tree and checks if state_index was visited previously in the episode"""
  if state_index == episode.state_idx and action_key == episode.action:
      return True
  if episode.prev_state_obj is None:
      return False
  return is_state_action_visited_previously(state_index, action_key, episode.prev_state_obj)
