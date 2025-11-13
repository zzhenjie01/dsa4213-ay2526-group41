"""
Contains the Multi-Armed Bandit algorithm classes.

Key change:
- All algorithms now accept a `numpy.random.Generator` instance (`rng`)
  for reproducible, encapsulated randomness.
"""

import numpy as np

# -----------------------------------------------
# Bandit Algorithm Classes
# -----------------------------------------------

class BaseBandit:
    """
    Base class for a Multi-Armed Bandit algorithm.
    
    Parameters:
    - k_arms (int): The number of arms (retrievers) in the bandit.
    - rng (np.random.Generator): A NumPy random number generator instance.
    """
    def __init__(self, k_arms, rng: np.random.Generator):
        if k_arms <= 0:
            raise ValueError("Number of arms must be greater than 0")
        self.k_arms = k_arms
        self.rng = rng
        self.reset()

    def select_arm(self):
        """Selects an arm to pull based on the algorithm's policy."""
        raise NotImplementedError

    def update(self, arm_index, reward):
        """Updates the algorithm's knowledge based on the reward received from the selected arm."""
        raise NotImplementedError

    def reset(self):
        self.timesteps = 0
        # Q-values: Estimated average reward for each arm
        self.Q = np.zeros(self.k_arms)
        # N: Number of times each arm has been pulled
        self.N = np.zeros(self.k_arms, dtype=int)


class EpsilonGreedy(BaseBandit):
    """
    Epsilon-Greedy MAB Algorithm.
    
    Parameters:
    - k_arms (int): The number of arms.
    - rng (np.random.Generator): Random number generator.
    - epsilon (float): The probability of exploring (choosing a random arm).
                       Must be between 0 and 1.
    """
    def __init__(self, k_arms, rng: np.random.Generator, epsilon: float):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1")
        self.epsilon = epsilon
        super().__init__(k_arms, rng)
        
    def select_arm(self):
        """
        With probability epsilon, choose a random arm (explore).
        With probability 1-epsilon, choose the arm with the highest Q-value (exploit).
        """
        self.timesteps += 1
        if self.rng.random() < self.epsilon:
            # Explore
            return self.rng.integers(0, self.k_arms)
        else:
            # Exploit
            # If multiple arms have the same max Q-value, np.argmax returns the first one.
            # We can break ties randomly if needed, but this is simpler.
            return np.argmax(self.Q)

    def update(self, arm_index, reward):
        """
        Update the Q-value for the pulled arm using an incremental average.
        Q_new = Q_old + (1/N) * (reward - Q_old)
        """
        self.N[arm_index] += 1
        self.Q[arm_index] += (1.0 / self.N[arm_index]) * (reward - self.Q[arm_index])

    def __str__(self):
        return f"EpsilonGreedy(e={self.epsilon})"


class UCB(BaseBandit):
    """
    Upper Confidence Bound (UCB1) MAB Algorithm.
    
    Parameters:
    - k_arms (int): The number of arms.
    - rng (np.random.Generator): Random number generator.
    - c (float): The exploration parameter. A higher 'c' encourages more exploration.
                 A common value is 2.
    """
    def __init__(self, k_arms, rng: np.random.Generator, c: float):
        if c < 0:
            raise ValueError("Exploration parameter 'c' must be non-negative")
        self.c = c
        super().__init__(k_arms, rng)

    def select_arm(self):
        """
        Selects the arm that maximizes the UCB score: Q(a) + c * sqrt(ln(t) / N(a))
        't' is the current total timestep.
        """
        self.timesteps += 1

        # First, check if any arm has not been pulled (N(a) == 0)
        # If so, pull one of those arms to initialize it.
        untried_arms = np.where(self.N == 0)[0]
        if len(untried_arms) > 0:
            return untried_arms[0]
        
        # If all arms have been tried at least once, calculate UCB scores
        # We add a small epsilon to the denominator to avoid division by zero
        # in a theoretical case, although our check above handles it.
        epsilon_denom = 1e-6
        ucb_scores = self.Q + self.c * np.sqrt(
            np.log(self.timesteps) / (self.N + epsilon_denom)
        )
        return np.argmax(ucb_scores)

    def update(self, arm_index, reward):
        """Update the Q-value for the pulled arm."""
        self.N[arm_index] += 1
        self.Q[arm_index] += (1.0 / self.N[arm_index]) * (reward - self.Q[arm_index])

    def __str__(self):
        return f"UCB(c={self.c})"


class ThompsonSampling(BaseBandit):
    """
    Thompson Sampling (for Bernoulli rewards) MAB Algorithm.
    Models the reward probability of each arm as a Beta distribution.
    
    Parameters:
    - k_arms (int): The number of arms.
    - rng (np.random.Generator): Random number generator.
    """
    def __init__(self, k_arms, rng: np.random.Generator):
        super().__init__(k_arms, rng)
        self.reset() # Base reset is fine, but we add alpha/beta
        
    def reset(self):
        """
        Resets the bandit's state.
        Alpha and Beta are the parameters for the Beta distribution.
        We initialize to 1 (Beta(1,1)), which is a uniform prior.
        """
        super().reset()
        # Alpha: counts of successes (reward=1)
        self.alpha = np.ones(self.k_arms)
        # Beta: counts of failures (reward=0)
        self.beta = np.ones(self.k_arms)

    def select_arm(self):
        """
        Draw a sample from each arm's current Beta(alpha, beta) distribution.
        Select the arm that produced the highest sample.
        """
        self.timesteps += 1
        # Draw samples
        samples = [
            self.rng.beta(self.alpha[i], self.beta[i]) 
            for i in range(self.k_arms)
        ]
        return np.argmax(samples)

    def update(self, arm_index, reward):
        """
        Update the alpha or beta parameter for the pulled arm.
        Assumes reward is binary (1 for success, 0 for failure).
        Thompson Sampling doesn't use Q-values and N directly.
        """
        # We discretize the reward: > 0.5 is a "success"
        reward_binary = 1 if reward > 0.5 else 0
        
        if reward_binary == 1:
            self.alpha[arm_index] += 1
        else:
            self.beta[arm_index] += 1

    def __str__(self):
        return "ThompsonSampling"
