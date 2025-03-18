import re, string, os
from typing import List, Union, Literal
from enum import Enum
from src.agent.environment import ActiveLearningEnv
import os
import tiktoken
from src.agent.prompts import *
from src.utils.ezr import *
from src.utils.helper import rows_to_markdown
from src.llm import load_model
import numpy as np



class LeapfrogLLM:
    """Wrapper for LLM to handle prompt engineering and reflection"""
    def __init__(self, llm_model, reflection_model=None):
        self.llm = llm_model
        self.reflection_model = reflection_model if reflection_model else llm_model
        self.reflections = []
        # Instead of tiktoken, use a simple character count estimate
        self.token_estimation_ratio = 3.5  # ~3.5 characters per token on average
    
    def estimate_tokens(self, text):
        """Estimate token count from text length"""
        return int(len(text) / self.token_estimation_ratio)
    
    def check_prompt_length(self, prompt, max_tokens=8000):
        """Check if prompt is too long for the model"""
        est_tokens = self.estimate_tokens(prompt)
        if est_tokens > max_tokens:
            # Truncate the history in the prompt
            return True
        return False
    
    def generate_better_point(self, selected_rows, history, reflect=True):
        """Generate a better point based on selected rows and history"""
        prompt = self._build_prompt(selected_rows, history)
        
        # Check if prompt is too long and truncate if necessary
        if self.check_prompt_length(prompt):
            # Use fewer history items in the prompt
            truncated_history = history[-3:] if len(history) > 3 else history
            prompt = self._build_prompt(selected_rows, truncated_history)
        
        response = self.llm.invoke(prompt).content
        
        # If reflection is enabled and we have history
        if reflect and history:
            reflection_prompt = self._build_reflection_prompt(selected_rows, history, response)
            
            # Check if reflection prompt is too long
            if self.check_prompt_length(reflection_prompt):
                truncated_history = history[-3:] if len(history) > 3 else history
                reflection_prompt = self._build_reflection_prompt(selected_rows, truncated_history, response)
            
            reflection = self.reflection_model.invoke(reflection_prompt).content
            self.reflections.append(reflection)
            
            # Use the reflection to improve the next generation
            improved_prompt = self._build_improved_prompt(selected_rows, history[-3:] if len(history) > 3 else history, reflection)
            response = self.llm.invoke(improved_prompt).content
        
        return self._parse_response(response)
    
    def _build_prompt(self, selected_rows, history):
        """Build the prompt for the LLM"""
        prompt = f"""
        You are assisting with active learning for tabular data. 
        
        I have randomly selected 4 data points from my dataset:
        {self._format_rows(selected_rows)}
        
        """
        
        if history:
            # Include the most recent history entries first (most relevant)
            prompt += f"""
            Previously, I've evaluated these points (ordered from oldest to newest):
            {self._format_history(history)}
            """
        
        prompt += """
        Please generate a new data point that you believe would be better than the randomly selected points.
        The goal is to minimize the "distance to heaven" metric. Lower values are better.
        
        Return your answer as a numpy array in the format: [feature1, feature2, ..., featureN]
        """
        
        return prompt
    
    def _build_reflection_prompt(self, selected_rows, history, previous_response):
        """Build a prompt for reflection"""
        prompt = f"""
        Let's reflect on our previous point generation.
        
        The randomly selected points were:
        {self._format_rows(selected_rows)}
        
        The point I generated was:
        {previous_response}
        
        Looking at the history of points and their scores:
        {self._format_history(history)}
        
        What patterns do you notice in the better-performing points?
        What features seem to correlate with lower "distance to heaven" scores?
        What could we do to generate an even better point next time?
        
        Please provide a thoughtful reflection that can guide our next point generation.
        """
        return prompt
    
    def _build_improved_prompt(self, selected_rows, history, reflection):
        """Build an improved prompt incorporating reflection"""
        prompt = f"""
        You are assisting with active learning for tabular data. 
        
        I have randomly selected 4 data points from my dataset:
        {self._format_rows(selected_rows)}
        
        Based on our previous attempts, here's what we've learned:
        {reflection}
        
        Previously, I've evaluated these points:
        {self._format_history(history)}
        
        Please generate a new data point that incorporates these reflections.
        The goal is to minimize the "distance to heaven" metric. Lower values are better.
        
        Return your answer as a numpy array in the format: [feature1, feature2, ..., featureN]
        """
        return prompt
    
    def _format_rows(self, rows):
        """Format rows for display in prompt"""
        result = ""
        for i, row in enumerate(rows):
            result += f"Point {i+1}: {row.tolist()}\n"
        return result
    
    def _format_history(self, history):
        """Format history for display in prompt"""
        result = ""
        for i, (input_rows, output_point, score) in enumerate(history):
            result += f"Iteration {i+1}:\n"
            result += f"  Generated point: {output_point.tolist()}\n"
            result += f"  Score: {score}\n"
        return result
    
    def _parse_response(self, response):
        """Parse the LLM's response into a numpy array"""
        try:
            # Use regex to extract array-like structures
            array_pattern = r'\[([^\]]+)\]'
            match = re.search(array_pattern, response)
            
            if match:
                array_text = match.group(1)
                # Clean up the array text
                values = [float(x.strip()) for x in array_text.split(',')]
                return np.array(values)
            else:
                # Fall back to basic extraction
                response = response.strip()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                if numbers:
                    return np.array([float(n) for n in numbers])
                else:
                    raise ValueError("No numbers found in response")
        except:
            # Return random values if parsing fails
            print("Failed to parse response, using random values")
            feature_count = len(self.reflections[0].split(',')) if self.reflections else 5
            return np.random.rand(feature_count)


class LeapfrogAgent:
    """Agent that uses the leapfrog method for active learning"""
    def __init__(self, env, llm, max_iterations=15, early_stop_threshold=0.01, patience=3):
        self.env = env
        self.llm = llm
        self.max_iterations = max_iterations
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience  # Number of iterations to wait for improvement
        self.history = []
        self.best_scores = []
    
    def run(self):
        """Run the leapfrog algorithm"""
        observation = self.env.reset()
        iterations_without_improvement = 0
        
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            
            # Generate a better point using the LLM
            action = self.llm.generate_better_point(
                self.env.dataset[np.random.choice(self.env.n, 4, replace=False)],
                self.history,
                reflect=(i > 0)  # Only reflect after the first iteration
            )
            
            # Take a step in the environment
            next_observation, reward, done, info = self.env.step(action)
            
            # Store the result
            self.history.append((info['selected_rows'], action, -reward))
            self.best_scores.append(self.env.best_score)
            
            # Render the environment
            self.env.render()
            
            # Check for early stopping
            if i > 0 and self.best_scores[-1] >= self.best_scores[-2]:
                iterations_without_improvement += 1
                print(f"No improvement for {iterations_without_improvement} iterations")
            else:
                iterations_without_improvement = 0
                print("New best score achieved!")
            
            if iterations_without_improvement >= self.patience:
                print(f"Early stopping triggered after {self.patience} iterations without improvement")
                break
            
            # Update observation
            observation = next_observation
            
            if done:
                break
        
        print("\n=== Final Results ===")
        print(f"Best point found: {self.env.best_point}")
        print(f"Best score achieved: {self.env.best_score}")
        print(f"Total iterations: {self.env.current_iteration}")
        
        return self.env.best_point, self.env.best_score
    
    def _should_early_stop(self):
        """Check if we should stop early"""
        if len(self.best_scores) < self.patience:
            return False
        
        # Check if there's been no improvement for 'patience' iterations
        return all(self.best_scores[-i-1] >= self.best_scores[-i-2] 
                   for i in range(self.patience-1))





# Example usage
def agents(args):
    # Create a synthetic dataset
    i = DATA(csv(args.dataset))
    
    llm_wrapper = LeapfrogLLM(
        llm_model = load_model(args, name = args.llm).get_pipeline()
    )
    
    # Create the environment
    env = ActiveLearningEnv(
        i=i,
        llm=llm_wrapper,
        max_iterations=15
    )
    
    # Create and run the agent
    agent = LeapfrogAgent(env, llm_wrapper)
    best_point, best_score = agent.run()
    
    print(f"Final best point: {best_point}")
    print(f"Final best score: {best_score}")

