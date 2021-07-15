"""Agents interaction with time period step environments."""
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import gamma


class RandomAgent:
    """Agent taking a random action each timestep.

    Args:
        env: the env the agent will be run on (so that the agent knows the actions to choose from)
    """

    def __init__(self, env: Any):
        self.action_space = env.action_space

    def act(self, previous_observation: np.ndarray) -> Tuple[Any, dict]:
        """Choose random action."""
        return self.action_space.sample(), {}


class EpsilonGreedy:
    """Agent greedily taking the highest conversion rate action (most of the time).

    Args:
        env: the env the agent will be run on (so that the agent knows the actions to choose from)
        epsilon: choose a random action `epsilon` percent of the time
    """

    def __init__(self, env: Any, epsilon=0.05):
        self.action_space = env.action_space
        # Dict of actions mapped to list of [num_trails, num_successes] for action
        self.conversion_counts: Dict[Any, List[int]] = {
            action: [0, 0] for action in range(self.action_space.n)
        }
        self.previous_action = None
        self.epsilon = epsilon

    def act(self, previous_observation: np.ndarray) -> Tuple[Any, dict]:
        """Choose highest conversion rate action (with some exploration)."""
        n_conversions = previous_observation[1]
        if self.previous_action is not None:
            self.conversion_counts[self.previous_action][0] += 1
            self.conversion_counts[self.previous_action][1] += n_conversions

        conversion_rate = {
            # Set conversion rate to infinity for unchosen actions
            # to ensure all actions are tested at least once
            action: n_conv / n_chosen if n_chosen != 0 else np.inf
            for action, (n_chosen, n_conv) in self.conversion_counts.items()
        }
        best_action = max(conversion_rate, key=lambda k: conversion_rate[k])

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            action = best_action

        self.previous_action = action
        return action, {}


# A list containing a tuple of (num_successes, timestep) for each time an action was picked,
# representing the full history of that action's conversions
ConversionHistory = List[Tuple[int, int]]


class BasePeriodic:
    """Base class for agents that optimises on separate conversion rates for each step in period.

    Args:
        env: the env the agent will be run on (so that the agent knows the actions to choose from)
        period_length: the length defining a repeating period, where each step will have a separate
            conversion rate. eg for a weekly periodicity, where each weekday has its own conversion
            rate, use period_length=7
    """

    def __init__(self, env: Any, period_length: int):
        self.action_space = env.action_space
        # List of dicts of actions mapped to their conversion history, for each action period idx
        self.conversions: List[Dict[Any, ConversionHistory]] = [
            {action: [] for action in range(self.action_space.n)} for _ in range(period_length)
        ]
        # Set conversion rate to infinity for unchosen actions to ensure all actions are tested at
        # least once
        self.upper_confidence_bounds = [
            {action: np.inf for action in range(self.action_space.n)} for _ in range(period_length)
        ]
        self.previous_action: Any = None
        self.step_number = 0
        self.period_length = period_length

    def compute_UCB_gamma(
        self,
        conversions: ConversionHistory,
        prior_alpha: float = 1.0,
        prior_beta: float = 0.0001,
        ucb_percentile: float = 0.95,
    ) -> float:
        """Compute Bayesian update on Gamma dist with priors and compute upper percentile value."""
        raise NotImplementedError()

    def act(self, previous_observation: np.ndarray) -> Any:
        """Choose action with highest upper confidence bound for step in period.

        Also use previous observation to update upper confidence bound for previous step in period.
        """
        current_idx = self.step_number % self.period_length  # current index within period length
        prev_idx = (self.step_number - 1) % self.period_length

        if self.previous_action is not None:
            obs, timestep = previous_observation[1], self.step_number - 1
            self.conversions[prev_idx][self.previous_action].append((obs, timestep))

            self.upper_confidence_bounds[prev_idx][self.previous_action] = self.compute_UCB_gamma(
                self.conversions[prev_idx][self.previous_action]
            )

        best_action = max(
            self.upper_confidence_bounds[current_idx],
            key=lambda k: self.upper_confidence_bounds[current_idx][k],
        )

        self.previous_action = best_action
        self.step_number += 1

        agent_info = {
            "ucb_selected_action": self.upper_confidence_bounds[current_idx][best_action]
        }
        return best_action, agent_info


class Periodic(BasePeriodic):
    """Agent that maintains and optimises on separate conversion rates for each step in period.

    Args:
        env: the env the agent will be run on (so that the agent knows the actions to choose from)
        period_length: the length defining a repeating period, where each step will have a separate
            conversion rate. eg for a weekly periodicity, where each weekday has its own conversion
            rate, use period_length=7
    """

    def compute_UCB_gamma(
        self,
        conversions: ConversionHistory,
        prior_alpha: float = 1.0,
        prior_beta: float = 0.0001,
        ucb_percentile: float = 0.95,
    ) -> float:
        """Compute Bayesian update on Gamma dist with priors and compute upper percentile value."""
        total_trials = len(conversions)
        total_successes = sum([c[0] for c in conversions])
        alpha = prior_alpha + total_successes
        beta = prior_beta + total_trials
        return gamma.ppf(ucb_percentile, alpha, scale=(1 / beta))
