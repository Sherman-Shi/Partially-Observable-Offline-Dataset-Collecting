import gymnasium as gym 
import numpy as np
import torch

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def sample(self):
        """Sample an action randomly."""
        action = self.action_space.sample()
        return torch.tensor(action, dtype=torch.float32)

    def log_prob(self, action):
        """Calculate the log probability of the action; adjust for the type of action space."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            # For discrete action spaces, the log probability of any action is uniform
            log_prob = np.log(1.0 / self.action_space.n)
        elif isinstance(self.action_space, gym.spaces.Box):
            # For continuous action spaces, calculate as before (uniform distribution assumed)
            total_actions = np.prod(self.action_space.high - self.action_space.low)
            log_prob = np.log(1.0 / total_actions)
        else:
            raise NotImplementedError("Action space type not supported")
        return log_prob

class BasicModel:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.policy = RandomPolicy(self.env.action_space)

    def load_policy(self):
        """Load the policy. Currently, it only loads the random policy."""
        return self.policy
