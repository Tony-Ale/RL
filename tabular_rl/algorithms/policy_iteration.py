import numpy as np
from algorithms.helpers import *
from environments.grid_world import GridworldEnv

def policy_evaluation(env: GridworldEnv, Vs:np.ndarray, policy:dict, gamma=0.5, threshold=1e-10, max_iters=100):
  for i in range(max_iters):
    delta = 0
    for state_index, action in policy.items():
      state = env.index_to_state(state_index)

      # handling terminal state
      if action is None:
        Vs[state] = 0.0
        continue

      Vo = Vs[state]
      next_state, reward, done, _ = env.transition(state, action)
      Vs_new = 1 * (reward + gamma*Vs[next_state])
      Vs[state] = Vs_new
      delta = max(delta, abs(Vo - Vs[state]))
    if delta < threshold:
      break
  return Vs

def policy_improvement(env:GridworldEnv, Vs:np.ndarray, policy:dict, gamma=0.5):
  policy_stable = True
  for state_index, action in policy.items():
    state = env.index_to_state(state_index)
    # handling terminal states
    if action is None:
      continue
    old_action = action

    action_value = np.ones(len(env.ACTIONS)) * float('-inf')
    for action in env.ACTIONS.keys():
      next_state, reward, done, _ = env.transition(state, action)
      action_value[action] = 1 * (reward + gamma*Vs[next_state])
    new_action = np.argmax(action_value)
    policy[state_index] = new_action
    if old_action != new_action:
      policy_stable = False
  return policy, policy_stable

def policy_iteration(env:GridworldEnv, policy:dict, gamma=0.95, threshold=1e-10, max_iters=100):
  # Initialization
  Vs = initialize_state_values(env)

  # Policy evaluation and improvement
  Vs_history = []
  Vs_history.append(Vs.copy().flatten())
  policy_stable = False
  while not policy_stable:
    Vs = policy_evaluation(env, Vs, policy, gamma, threshold, max_iters)
    policy, policy_stable = policy_improvement(env, Vs, policy, gamma)
    Vs_history.append(Vs.copy().flatten())
  return policy, Vs, Vs_history
