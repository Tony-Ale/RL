from algorithms.helpers import initialize_policy, generate_episode, State_Data
from environments.grid_world import GridworldEnv
import numpy as np
import random

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

def montecarlo_es_estimating_policy(env:GridworldEnv, gamma=0.9, max_iters=5000):
    """Estimating policy using monte carlo exploring starts, every first visit to state-action pair is used"""
    # Initialize policy, returns and action-value function
    policy = initialize_policy(env)
    returns = initialize_returns(env)
    Q_s_a = initialize_action_value(env)

    state_action_pairs = list(Q_s_a.keys())
    for _ in range(max_iters):
        So, Ao = random.choice(state_action_pairs)
        
        episode = generate_episode(env, policy, start_state_idx=So, start_action_idx=Ao)
        G = 0

        while episode is not None:
            G = gamma*G + episode.reward
            S_t = episode.state_idx
            A_t = episode.action

            S_t_1_data = episode.prev_state_obj
            if S_t_1_data is None or not is_state_action_visited_previously(S_t, A_t, S_t_1_data):
                key = generate_state_action_keys(S_t, A_t)
                returns[key].append(G)
                Q_s_a[key] = np.mean(returns[key])
                
                S_t_actions = []
                action_values = []
                for action_key in env.ACTIONS:
                    action_val = Q_s_a[generate_state_action_keys(S_t, action_key)]
                    action_values.append(action_val)
                    S_t_actions.append(action_key) 
                max_index = np.argmax(action_values)
                policy[S_t] = S_t_actions[max_index]

            episode = episode.prev_state_obj
    return policy
