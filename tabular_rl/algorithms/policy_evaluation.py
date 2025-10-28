import numpy as np 
from algorithms.helpers import initialize_state_values
from environments.grid_world import GridworldEnv

def generate_policy(env:GridworldEnv):
    """The functions generates a policy using a uniform distribution
        The index of the numpy array corresponds to the actions
    """
    policy = {}

    state_indices = [idx for idx in range(env.width * env.height) if env.index_to_state(idx) not in env.goals]

    for state_index in state_indices:
        policy[state_index] = np.ones(len(env.ACTIONS))/len(env.ACTIONS)
    
    return policy

def policy_evaluation(env:GridworldEnv, policy:dict, gamma=0.9, max_iters=100, threshold=1e-5):
    Vs = initialize_state_values(env)
    state_indices = [idx for idx in range(env.width * env.height) if env.index_to_state(idx) not in env.goals]
    for _ in range(max_iters):
        delta = 0

        for state_index in state_indices:
            state = env.index_to_state(state_index)
            Vo = Vs[state]
            value = 0
            for action, Pi_a in enumerate(policy[state_index]):
                new_state, reward, _, _ = env.transition(state, action)
                value += Pi_a * (1 * (reward + gamma*Vs[new_state]))

                delta = max(delta, abs(Vo -Vs[state]))
                
            Vs[state] = value
        
        if delta < threshold:
            break
    return Vs


