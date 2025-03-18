import gym
import numpy as np
from gym import spaces
from src.utils.ezr import *


class ActiveLearningEnv(gym.Env):
    def __init__(self, i, llm, max_iterations=15):
        super(ActiveLearningEnv, self).__init__()
        self.i = i
        self.dataset = np.array(i.rows)
        self.n = len(i.rows)
        self.llm = llm
        self.selected_points = []  # Stores (input, output, score)
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.best_score = float('inf')
        self.best_point = None
        
        # Define feature dimensionality from dataset
        self.feature_dim = self.dataset.shape[1]
        
        # Action space: Define the space for generating a better point
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,))
        
        # Observation space: Dataset + past results
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, self.feature_dim))
    
    def step(self, action):
        # Process the action (in this case, action is the LLM's response)
        # The action here is the textual response from the LLM
        
        # Randomly select 4 rows from the dataset
        selected_indices = np.random.choice(self.n, 4, replace=False)
        
        
        # Process the LLM's response to extract the better point
        better_point = self._parse_llm_response(action)
        
        # Evaluate using distance_to_heaven
        score = d2h(self.dataset,better_point)
        
        # Store result
        self.selected_points.append((better_point, score))
        
        # Update best score and point if needed
        if score < self.best_score:
            self.best_score = score
            self.best_point = better_point
        
        # Reward: Negative of the distance (lower is better)
        # We could also use improvement over previous best
        reward = -score
        
        # Alternative reward: Improvement over previous best
        # if len(self.selected_points) > 1:
        #     prev_score = self.selected_points[-2][2]
        #     reward = prev_score - score  # Positive if improved, negative if worsened
        
        # Increment iteration counter
        self.current_iteration += 1
        
        # Done condition
        done = self.current_iteration >= self.max_iterations
        
        # Information dictionary
        info = {
            'iteration': self.current_iteration,
            'score': score,
            'best_score': self.best_score,
            'selected_rows': selected_rows
        }
        
        # Current state combines dataset and history
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def reset(self):
        self.selected_points = []
        self.current_iteration = 0
        self.best_score = float('inf')
        self.best_point = None
        return self._get_observation()
    
    def _get_observation(self):
        # Combine dataset and history into an observation
        # This could be formatted as a prompt for the LLM
        return self.dataset
    
    def _parse_llm_response(self, response):
        # Parse the LLM's textual response into a data point
        # This is a placeholder - you'll need to implement actual parsing
        try:
            # Assuming response is a string that can be parsed into a numpy array
            # This will depend on your LLM's output format
            return np.array(eval(response))
        except:
            # If parsing fails, return a random point as fallback
            print("Failed to parse LLM response, using random point")
            return np.random.rand(self.feature_dim)
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Iteration: {self.current_iteration}/{self.max_iterations}")
            print(f"Current best score: {self.best_score}")
            print(f"Best point: {self.best_point}")
            print(f"Total points evaluated: {len(self.selected_points)}")
        return None
    
    def is_terminated(self):
        return self.current_iteration >= self.max_iterations
    
    def is_truncated(self):
        # Implement any additional termination conditions
        return False
    
    def is_correct(self):
        # Define what makes a solution "correct"
        # For example, if score is below a threshold
        threshold = 0.1  # Adjust based on your problem
        return self.best_score < threshold