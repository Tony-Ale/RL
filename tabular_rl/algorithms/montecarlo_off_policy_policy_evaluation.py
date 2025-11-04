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

def montecarlo_off_policy_policy_evaluation(env:GridworldEnv, target_policy:dict, gamma=0.95, num_episodes=5000, is_target_policy_stochastic=False):
    """
    Perform Monte Carlo Off-Policy Policy Evaluation for estimating Q (action value).

    Args:
        env: The environment to evaluate the policy on.
        target_policy: Deterministic or stochastic policy to be evaluated.
        num_episodes: Number of episodes to sample.
    """

    Q_s_a = initialize_action_value(env)
    C_s_a = initialize_state_action_cumulative_sum(env)

    for _ in range(num_episodes):
        behaviour_policy = generate_behaviour_policy(env)

        episode = generate_episode(env, behaviour_policy, is_policy_stochastic=True)
        G = 0
        W = 1

        while episode.prev_state_obj is not None and W != 0:
            G = gamma*G + episode.reward
            St = episode.state_idx
            At = episode.action
            key = generate_state_action_keys(St, At)
            C_s_a[key] += W 
            Q_s_a[key] += (W/C_s_a[key])*(G - Q_s_a[key])
            
            if is_target_policy_stochastic:
                W = W * target_policy[St][At]/max(behaviour_policy[St][At], 1e-8)
            else:
                W = W * (1 if target_policy[St] == At else 0)/max(behaviour_policy[St][At], 1e-8)

            episode = episode.prev_state_obj
    return Q_s_a


    