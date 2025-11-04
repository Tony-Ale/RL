import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class GridworldEnv:
  """
  Minimal deterministic Gridworld:
    - grid with coordinates (r, c) where 0 <= r < height, 0 <= c < width
    - actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    - <0, 0> is at the top left corner
    - step returns: (next_state, reward, done, info)
  """
  VERBOSE_ACTIONS = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
  ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

  def __init__(self, width=4, height=4, start=(0, 0), goals=[(3, 3), ],
                step_reward=-0.05, goal_reward=1.0, obstacles=None, seed=None):
    self.width = width
    self.height = height
    self.start = tuple(start)
    self.goals = goals
    self.step_reward = step_reward
    self.goal_reward = goal_reward
    self.obstacles = set(obstacles or [])
    self.rng = np.random.default_rng(seed)
    self.reset()

  def state_to_index(self, state):
    r, c = state
    return r * self.width + c

  def index_to_state(self, idx):
    return (idx // self.width, idx % self.width)

  def reset(self):
    """Reset environment and return initial state (row, col)."""
    self.pos = tuple(self.start)
    self.done = False
    return self.pos

  def step(self, action):
    new_state, reward, done, info = self.transition(self.pos, action)
    self.pos = new_state
    self.done = done
    return new_state, reward, done, info

  def transition(self, state:tuple, action:int):
    """
    Take an action (int). Returns (next_state, reward, done, info).
    - If action would move into an obstacle or outside grid, agent stays.
    - Terminal when agent reaches goal (then done=True and no further movement).
    """
    if state in self.goals:
      return state, 0.0, True, {"msg": "already_terminal"}

    dr, dc = self.ACTIONS[int(action)]
    r, c = state
    nr, nc = r + dr, c + dc

    # Bound check
    nr = max(0, min(self.height - 1, nr))
    nc = max(0, min(self.width - 1, nc))

    attempted = (nr, nc)
    if attempted in self.obstacles:
        # Can't move into obstacle: stay in place
        nr, nc = r, c

    new_state = (nr, nc)
    done = False
    info = {"attempted": attempted, "pos_index": self.state_to_index(new_state)}

    if new_state in self.goals:
        reward = self.goal_reward
        done = True
        info["is_terminal"] = True
    else:
        reward = self.step_reward
        info["is_terminal"] = False

    return new_state, reward, done, info

  def render(self, ax=None):
    """
    Render a simple grid: agent as circle, goal as square, obstacles as black cells.
    Returns the matplotlib Axes used (so external code can plt.show()).
    """
    grid = np.zeros((self.height, self.width))
    fig_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    # draw obstacles as 1, goal as 0.5 for visual cue (but we will annotate)
    for (orow, ocol) in self.obstacles:
        grid[orow, ocol] = 1.0

    ax.imshow(grid, interpolation='nearest', origin='upper')
    # grid lines
    ax.set_xticks(np.arange(-0.5, self.width, 1), minor=False)
    ax.set_yticks(np.arange(-0.5, self.height, 1), minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Agent
    r, c = self.pos
    ax.scatter([c], [r], s=200, marker='o', facecolors='none', edgecolors='blue', linewidths=2, label='agent')
    # Goal
    for goal in self.goals:
      gr, gc = goal
      ax.scatter([gc], [gr], s=200, marker='s', facecolors='none', edgecolors='green', linewidths=2, label='goal')
    ax.set_title(f"Gridworld: pos={self.pos}, goals={self.goals}")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    if not fig_provided:
        plt.show()
    return ax
