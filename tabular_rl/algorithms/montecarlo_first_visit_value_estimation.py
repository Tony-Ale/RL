from environments.grid_world import GridworldEnv
from algorithms.helpers import initialize_state_values, initialize_policy, generate_episode, State_Data
import numpy as np

def initialize_returns(env:GridworldEnv)->dict[int, list]:
    returns = {}

    for state_index in range(env.width * env.height):
        returns[state_index] = []
    return returns

def is_state_visited_previously(state_index, episode:State_Data):
  """moves down the tree and checks if state_index was visited previously in the episode"""
  if state_index == episode.state_idx:
      return True
  if episode.prev_state_obj is None:
      return False
  return is_state_visited_previously(state_index, episode.prev_state_obj)

def montecarlo_first_visit_value_estimation(env:GridworldEnv, gamma=0.9, max_iters=5000):
    Vs = initialize_state_values(env)
    returns = initialize_returns(env)
    policy = initialize_policy(env, use_good_policy=True)

    for _ in range(max_iters):
        episode = generate_episode(env, policy)
        G = 0
        S_t_data = episode
        while S_t_data is not None:
            G = gamma*G + S_t_data.reward

            S_t = S_t_data.state_idx
            S_t_1_data = S_t_data.prev_state_obj

            if S_t_1_data is None or not is_state_visited_previously(S_t, S_t_1_data):
                returns[S_t].append(G)

                state = env.index_to_state(S_t)

                V_average = np.mean(returns[S_t])


                Vs[state] = V_average

            S_t_data = S_t_data.prev_state_obj

    return Vs



