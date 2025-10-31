from algorithms.helpers import generate_episode, State_Data
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

def initialize_epsilon_soft_policy(env:GridworldEnv, epsilon=0.2):
    """Creates an arbitrarily soft policy where one action is randomly selected as the greedy action"""
    policy = {}
    state_indices = [idx for idx in range(env.width*env.height) if env.index_to_state(idx) not in env.goals ]
    actions = env.ACTIONS
    for state_index in state_indices:
        pr_non_greedy = epsilon/len(actions) # probability of non-greedy actions
        pr_greedy = 1 - epsilon + pr_non_greedy
        greedy_index = np.random.randint(len(actions))
        probs = np.ones(len(actions)) * pr_non_greedy
        probs[greedy_index] = pr_greedy
        policy[state_index] = probs
    return policy


def montecarlo_epsilon_soft_policies(env:GridworldEnv, policy:dict, epsilon=0.2, gamma=0.9, max_iters=5000):
    """
    This function implements the Monte Carlo method for estimating a policy using 
    on-policy first-visit updates and ε-soft policies.
    """
    # Initialization
    Q_s_a = initialize_action_value(env)
    returns = initialize_returns(env)

    for _ in range(max_iters):
        episode = generate_episode(env, policy, is_policy_stochastic=True)
        G = 0

        while episode.prev_state_obj is not None:
            G = gamma * G + episode.reward
            S_t = episode.state_idx
            A_t = episode.action
            if episode.prev_state_obj is None or not is_state_action_visited_previously(S_t, A_t, episode.prev_state_obj):
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

                pr_non_greedy = epsilon / len(env.ACTIONS)
                probs = np.ones(len(env.ACTIONS)) * pr_non_greedy
                probs[S_t_actions[max_index]] = 1 - epsilon + pr_non_greedy

                policy[S_t] = probs

            episode = episode.prev_state_obj
    return policy
