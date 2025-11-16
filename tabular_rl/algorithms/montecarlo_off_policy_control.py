from algorithms.helpers import *
from environments.grid_world import GridworldEnv
import numpy as np

def generate_behaviour_policy(env:GridworldEnv):
    # Generates a stochastic behaviour policy
    policy = {}
    state_indices = [idx for idx in range(env.width * env.height) if env.index_to_state(idx) not in env.goals]

    for state_index in state_indices:
        policy[state_index] = np.random.dirichlet(np.ones(len(env.ACTIONS)))
    
    return policy

def initialize_state_action_cumulative_sum(env:GridworldEnv):

    C_s_a = {}
    state_indices = [idx for idx in range(env.width * env.height) if env.index_to_state(idx) not in env.goals]

    for state_index in state_indices:
        for action_idx in env.ACTIONS:
            key = generate_state_action_keys(state_index, action_idx)
            C_s_a[key] = 0
    return C_s_a

def get_max_action(env:GridworldEnv, Q_s_a:dict, state_index:int):
    values = []
    actions = []
    for action_idx in env.ACTIONS:
        key = generate_state_action_keys(state_index, action_idx)
        values.append(Q_s_a[key])
        actions.append(action_idx)

    values = np.array(values)
    max_value = np.max(values)
    max_indices = np.where(values == max_value)[0]

    # break ties consistently by choosing the highest action index
    chosen_idx = max_indices.max()
    chosen_action = actions[chosen_idx]
    return chosen_action

def generate_policy_from_action_value(env:GridworldEnv, Q_s_a:dict):
    """for each state get the action that has the highest value"""
    policy = {}

    state_indices = [idx for idx in range(env.width * env.height) if env.index_to_state(idx) not in env.goals]

    for state_index in state_indices:
        action = get_max_action(env, Q_s_a, state_index)
        policy[state_index] = action
    return policy

def montecarlo_off_policy_control(env:GridworldEnv, gamma=0.95, num_episodes=5000):
    """Estimating the target policy using monte carlo off policy method"""
    Q_s_a = initialize_action_value(env)
    C_s_a = initialize_state_action_cumulative_sum(env)
    policy = generate_policy_from_action_value(env, Q_s_a)

    for _ in range(num_episodes):
        b = generate_behaviour_policy(env)
        episode = generate_episode(env, b, is_policy_stochastic=True)
        G = 0
        W = 1

        while episode.prev_state_obj is not None and W != 0:
            G = gamma*G + episode.reward
            St = episode.state_idx
            At = episode.action
            key = generate_state_action_keys(St, At)
            C_s_a[key] = C_s_a[key] + W
            Q_s_a[key] = Q_s_a[key] + (W/C_s_a[key])*(G-Q_s_a[key])
            action = get_max_action(env, Q_s_a, St)
            policy[St] = action

            if At != action:
                break
            W = W/b[St][At]

            episode = episode.prev_state_obj
    return policy
