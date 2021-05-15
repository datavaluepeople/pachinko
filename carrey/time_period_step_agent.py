"""Agents interaction with time period step environments."""
from typing import Any, Tuple

import numpy as np


class RandomAgent:
    """Agent taking a random action each timestep.

    Args:
        action_space: a list of the possible actions in the environment
    """

    def __init__(self, action_space: list):
        self.action_space = action_space

    def act(self, previous_observation: np.array) -> Tuple[Any, dict]:
        """Choose random action."""
        return np.random.choice(self.action_space), {}


class EpsilonGreedy:
    """Agent greedily taking the highest conversion rate action (most of the time).

    Args:
        action_space: a list of the possible actions in the environment
        epsilon: choose a random action `epsilon` percent of the time
    """

    def __init__(self, action_space: list, epsilon=0.05):
        self.action_space = action_space
        self.conversion_counts = {action: [0, 0] for action in action_space}
        self.previous_action = None
        self.epsilon = epsilon

    def act(self, previous_observation: np.array) -> Tuple[Any, dict]:
        """Choose highest conversion rate action (with some exploration)."""
        n_conversions = previous_observation[1]
        if self.previous_action is not None:
            self.conversion_counts[self.previous_action][0] += 1
            self.conversion_counts[self.previous_action][1] += n_conversions

        conversion_rate = {
            # Set conversion rate to infinity for unchosen actions
            # to ensure all actions are tested at least once
            action: n_conv / n_chosen if n_chosen != 0 else np.inf
            for action, (n_conv, n_chosen) in self.conversion_counts.items()
        }
        best_action = max(conversion_rate, key=conversion_rate.get)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = best_action

        self.previous_action = action
        return action, {}
