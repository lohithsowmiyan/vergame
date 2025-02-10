import gym
from gym import spaces
import numpy as np
import itertools

class RowSelectionEnv(gym.Env):
    def __init__(self, dataset, labeled_indices):
        super(RowSelectionEnv, self).__init__()

        self.dataset = dataset
        self.labeled_indices = set(labeled_indices)
        self.unlabeled_indices = set(range(len(dataset))) - self.labeled_indices

        self.total_cost = 0  # Cost tracker
        self.max_cost = 20   # Max cost limit
        
        # Action space: Choosing 4 indices from 100 rows
        self.action_space = spaces.MultiDiscrete([len(dataset)] * 4)

        # Observation space: Entire dataset (could be feature vectors)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(dataset), dataset.shape[1]), dtype=np.float32)

        self.reset()

    def reset(self):
        self.total_cost = 0  # Reset cost
        return self.dataset  # Return the dataset as observation

    def step(self, action):
        selected_rows = set(action)  # Ensure unique row selections

        # Compute additional cost for choosing unlabeled rows
        unlabeled_selected = len([idx for idx in selected_rows if idx not in self.labeled_indices])
        cost = unlabeled_selected  # +1 per unlabeled row

        # If adding cost exceeds max_cost, terminate early with heavy penalty
        if self.total_cost + cost > self.max_cost:
            return self.dataset, -100, True, {"cost": self.total_cost}  # High penalty for exceeding cost

        self.total_cost += cost  # Update cost

        # Compute reward using another agent (dummy score function for now)
        reward = self.evaluate_selection(selected_rows)

        done = self.total_cost >= self.max_cost  # Terminate when max cost is reached
        return self.dataset, reward, done, {"cost": self.total_cost}

    def evaluate_selection(self, selected_rows):
        """Dummy function to score selected rows (replace with real scoring model)"""
        return np.random.rand() * 10  # Placeholder: replace with actual evaluation model

    def render(self, mode="human"):
        print(f"Total Cost: {self.total_cost}")

    def close(self):
        pass
