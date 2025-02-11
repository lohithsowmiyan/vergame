import gym
from gym import spaces
import numpy as np
import itertools
from src.utils.ezr import DATA,csv, chebyshev
from src.reward_model.models.warm_start import WARM_FEW_API

class RowSelectionEnv(gym.Env):
    def __init__(self, dataset, labeled_indices, args):
        super(RowSelectionEnv, self).__init__()

        self.dataset = np.array(dataset)
        self.labeled_indices = set(labeled_indices)
        self.unlabeled_indices = set(range(len(dataset))) - self.labeled_indices

        self.total_cost = 0  # Cost tracker
        self.max_cost = 20   # Max cost limit
        self.i = DATA(csv(args.dataset))
        self.cur_best = min([chebyshev(self.i, r) if _ in labeled_indices else 1 for _,r  in enumerate(dataset)])
        print("best initially", self.cur_best)
        
        # Action space: Choosing 4 indices from 100 rows
        self.action_space = spaces.MultiDiscrete([len(dataset)] * 4)

        # Observation space: Entire dataset (could be feature vectors)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(dataset), self.dataset.shape[1]), dtype=np.float32)

        self.reset()

    def reset(self):
        self.total_cost = 0  # Reset cost
        return self.dataset  # Return the dataset as observation

    def step(self, action):
        selected_rows = set(action)  # Ensure unique row selections




        # Compute additional cost for choosing unlabeled rows
        unlabeled_selected = len([idx for idx in selected_rows if idx not in self.labeled_indices])
        cost = unlabeled_selected  # +1 per unlabeled row\

        self.labeled_indices.update(selected_rows)


        # If adding cost exceeds max_cost, terminate early with heavy penalty
        if self.total_cost + cost > self.max_cost:
            return self.dataset, -100, True, {"cost": self.total_cost}  # High penalty for exceeding cost

        self.total_cost += cost  # Update cost

        # Compute reward using another agent (dummy score function for now)
        reward = self.evaluate_selection(selected_rows)

        done = self.total_cost >= self.max_cost  # Terminate when max cost is reached
        return self.dataset, reward, done, {"cost": self.total_cost} 

    def evaluate_selection(self, selected_rows, args):
        """function to score selected rows (replace with real scoring model)"""
        i = DATA(csv(args.dataset))

        done = [dataset[i] for i in selected_rows]
        done = set(tuple(_) for _ in done)
        todo = set((tuple(_) for _ in i.rows)) - done
        done, new_done, todo = WARM_FEW_API(i, args,  list(todo), list(done), method = 'LLM')

        if chebyshev(i, new_done[0]) < self.cur_best:
            return 10
        else:
            return -5






        

    def render(self, mode="human"):
        print(f"Total Cost: {self.total_cost}")

    def get_env_state(self):
        return self.dataset, self.labeled_indices

    def close(self):
        pass
