from algorithms.helpers import initialize_state_values
from environments.grid_world import GridworldEnv

def value_iteration(env:GridworldEnv, gamma=0.9, threshold=1e-10, max_iters=100):
    actions = env.ACTIONS
    Vs = initialize_state_values(env)

    for _ in range(max_iters):
        delta = 0

        for state_index in range(env.width * env.height):
            state = env.index_to_state(state_index)
            Vo = Vs[state]

            values = []
            for action in actions:
                next_state, reward, _, _= env.transition(state, action)

                # Value of state due to action
                V_s = 1 * (reward + gamma*Vs[next_state])
                values.append(V_s)
            V_s_max = max(values)
            Vs[state] = V_s_max
            delta = max(delta, abs(Vo - V_s_max))
        if delta < threshold:
            break
    
    # creating the policy 
    policy = {}
    terminal_states = [env.state_to_index(goal) for goal in env.goals] 
    for state_index in range(env.width * env.height):
        state = env.index_to_state(state_index)
        if state_index in terminal_states:
            policy[state_index] = None
        else:
            best_action = None 
            best_value = float('-inf')
            for action in actions:
                next_state, reward, _, _= env.transition(state, action)
                value = 1 * (reward + gamma*Vs[next_state])
                if value > best_value:
                    best_value = value
                    best_action = action
            policy[state_index] = best_action
    return policy, Vs
