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
    Empirical Gaussian Thompson Sampling MAB Algorithm.
    
    This algorithm is suited for continuous rewards (positive or negative),
    which fits the `reward = score - lambda * norm_cost` structure.
    
    It models the reward for each arm as a Normal (Gaussian) distribution
    by learning the empirical mean (self.Q) and empirical variance 
    (from self.sum_squared_rewards) of the rewards.
    
    It samples from this learned distribution to balance exploration/exploitation:
    - The mean of the sample is the arm's estimated average reward (exploitation).
    - The standard deviation (scale) of the sample is derived from the
      standard error of the mean, which shrinks as more data is collected
      (exploration).

    Parameters:
    - k_arms (int): The number of arms.
    - rng (np.random.Generator): Random number generator.
    """
    def __init__(self, k_arms, rng: np.random.Generator):
        # The base __init__ calls self.reset()
        super().__init__(k_arms, rng)
        
    def reset(self):
        """Resets the bandit's state, including Q, N, and sum_squared_rewards."""
        # Reset Q, N, and timesteps from the base class
        super().reset()
        # Add tracker for sum of squared rewards (for variance calculation)
        self.sum_squared_rewards = np.zeros(self.k_arms)
    
    def select_arm(self):
        """
        Draw a sample from each arm's learned Normal distribution and
        select the arm with the highest sample.
        """
        self.timesteps += 1
        samples = np.zeros(self.k_arms)
        
        for i in range(self.k_arms):
            # The mean of our belief is the current empirical mean
            mean = self.Q[i]
            
            # Calculate the standard error of the mean as our uncertainty (scale)
            if self.N[i] > 1:
                # Calculate empirical variance: Var(X) = E[X^2] - (E[X])^2
                # E[X^2] = sum_squared_rewards / N
                # E[X]   = Q
                variance = (self.sum_squared_rewards[i] / self.N[i]) - self.Q[i]**2
                
                # Clamp variance to a small positive number for numerical stability
                # This prevents sqrt(0) or sqrt(negative) from floating point errors
                variance = max(variance, 1e-6) 
                
                # Our uncertainty is in the mean, so we use standard error: sqrt(var / N)
                # Added a small constant to avoid zero std deviation
                std_error = np.sqrt((variance / self.N[i]) + 1e-8)
            else:
                # Not enough data to calculate variance (N=0 or N=1).
                # Use a large standard deviation to encourage initial exploration.
                std_error = 1.0 # Initial prior uncertainty
            
            # Draw a sample from N(mean, std_error)
            samples[i] = self.rng.normal(loc=mean, scale=std_error)
        
        # Select the arm with the highest sample ("optimism in the face of uncertainty")
        return np.argmax(samples)
    
    def update(self, arm_index, reward):
        """
        Update the Q-value (mean), N (count), and sum_squared_rewards
        for the pulled arm.
        """
        # 1. Update count (must be first)
        self.N[arm_index] += 1
        
        # 2. Update sum of squares (for variance calculation)
        self.sum_squared_rewards[arm_index] += reward ** 2
        
        # 3. Update mean (Q-value) using incremental average
        # Q_new = Q_old + (1/N) * (reward - Q_old)
        self.Q[arm_index] += (1.0 / self.N[arm_index]) * (reward - self.Q[arm_index])

    def __str__(self):
        return "ThompsonSampling(EmpiricalGaussian)"