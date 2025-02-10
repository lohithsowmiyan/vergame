import re, string, os
from typing import List, Union, Literal
from enum import Enum
from src.agent.environment import RowSelectionEnv
import os
import tiktoken
from src.agent.prompts import *
from src.utils.ezr import *
from src.utils.helper import rows_to_markdown
from src.llm import load_model



def format_step(text: str) -> str:
    """Simple formatter to strip extra whitespace from LLM responses."""
    return text.strip()

def format_reflections(reflections) -> str:
    """Join a list of reflection strings into one text block."""
    return "\n".join(reflections)



# =============================================================================
# LLM Interface Stubs
# =============================================================================
# (Replace these stubs with your actual LLM classes or imports.)
# For example, you might be using a class like OpenAI from your LLM library.
#
# from your_llm_library import OpenAI, BaseLLM, PromptTemplate

# For this example, we assume that calling the LLM object with a prompt returns a string.
# Example:
#    response = self.llm(prompt_text)
# where self.llm might be an instance of OpenAI with preconfigured parameters.

# =============================================================================
# RowSelectionReactAgent
# =============================================================================

class LeapFrogAgent:
    """
    A ReAct agent for a row selection task.
    
    The agent uses an LLM to generate a chain-of-thought (its "thoughts")
    and to decide on an action (e.g., which 4 rows to select). The LLM output is
    appended to a scratchpad. This agent runs in a loop until the environment
    signals termination or the prompt grows too long.
    """
    def __init__(self,
                 task_description: str,
                 env,  # An instance of your custom row selection environment.
                 agent_prompt: str = row_selection_agent_prompt,
                 react_llm=None,  # An LLM instance (e.g., OpenAI with text-davinci-003).
                 ) -> None:

        self.task_description = task_description
        self.agent_prompt = agent_prompt
        self.react_examples = ROW_SELECTION_EXAMPLES
        self.env = env
        self.env.reset()
        self.reset()  # Initialize the scratchpad and step counter.
        self.truncated = False
        self.reward = 0
        self.terminated = False

        # Initialize the reaction LLM if not provided.
        if react_llm is None:
            self.llm = load_model(args, name = 'gemini').get_pipeline()
        else:
            self.llm = react_llm

        # Initialize tokenizer for token counting.
        # self.enc = tiktoken.encoding_for_model("text-davinci-003")

    def run(self, reset=True) -> None:
        if reset:
            self.env.reset()
            self.reset()
        while not (self.is_truncated() or self.is_terminated()):
            self.step()

    def step(self) -> None:
        # ---- Act: Generate an action (e.g., the row indices to select).
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # ---- Observe: Execute the action in the environment.
        self.scratchpad += f'\nObservation {self.curr_step}: '
        # Assume env.step returns: observation, reward, terminated, truncated, curr_step.
        observation, self.reward, self.terminated, self.truncated, self.curr_step = self.env.step(action)
        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        prompt_text = self._build_agent_prompt()
        response = self.llm(prompt_text)
        return format_step(response)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            question=self.task_description,
            scratchpad=self.scratchpad
        )

    def is_terminated(self) -> bool:
        # Assumes the environment provides an is_terminated() method.
        return self.env.is_terminated()

    # def is_truncated(self) -> bool:
    #     # Checks if the environment is truncated or if the prompt is too long.
    #     prompt_length = len(self.enc.encode(self._build_agent_prompt()))
    #     return self.env.is_truncated() or (prompt_length > 3896)

    def reset(self) -> None:
        self.scratchpad = ''
        self.curr_step = 1

    def is_correct(self) -> bool:
        # Assumes the environment provides an is_correct() method to check the action quality.
        return self.env.is_correct()

class LeapFrogReflectionAgent(LeapFrogAgent):
    """
    A self-reflecting  agent for the row selection task.
    
    This agent uses a separate reflection LLM to reflect on its chain-of-thought
    when the environment terminates (or the prompt is too long) and the current
    action is not deemed correct.
    """
    def __init__(self,
                 task_description: str,
                 env,
                 agent_prompt: str = row_selection_reflect_agent_prompt,
                 reflect_prompt: str = row_selection_reflect_prompt,
                 react_llm=None,
                 reflect_llm=None,
                 ) -> None:
        super().__init__(task_description, env, agent_prompt, react_llm)
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = ROW_SELECTION_REFLECT_EXAMPLES
        self.reflections = []

        # Initialize the reflection LLM if not provided.
        if reflect_llm is None:
            self.llm = load_model(args, name = 'gemini').get_pipeline()
        else:
            self.llm = reflect_llm

    def run(self, reset=True) -> None:
        # If the episode is terminated/truncated and the current action is not correct,
        # perform a reflection step.
        if (self.is_terminated() or self.is_truncated()) and not self.is_correct():
            self.reflect()
        super().run(reset) 

    def reflect(self) -> None:
        reflection = self.prompt_reflection()
        self.reflections.append(reflection)
        print("Reflection:", reflection)

    def prompt_reflection(self) -> str:
        prompt_text = self._build_reflection_prompt()
        response = self.reflect_llm(prompt_text)
        return format_step(response)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.task_description,
            scratchpad=self._format_scratchpad()
        )

    def _build_agent_prompt(self) -> str:
        # Incorporate reflections into the agent prompt.
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=format_reflections(self.reflections),
            question=self.task_description,
            scratchpad=self.scratchpad
        )

    def _format_scratchpad(self) -> str:
        # Truncate or summarize the scratchpad if it grows too long.
        lines = self.scratchpad.split('\n')
        lines_by_tokens = sorted(lines, key=lambda x: len(self.enc.encode(x)))
        while len(self.enc.encode('\n'.join(lines))) > 1600:
            # Replace the longest line with a summary (here, we simply shorten it).
            ind = lines.index(lines_by_tokens.pop(-1))
            line = lines[ind]
            lines[ind] = line.split(':')[0] + ': ...'
        return '\n'.join(lines)



def run_agent(args):
    d = DATA(csv(args.dataset))
    tot_rows = random.choices(d.rows, k = 50)
    labelled_indices = random.choices(range(0,50), k = 10)

    #print(d.cols.names)

    x_len = len(tot_rows[0]) - len(d.cols.y)
    table = [
    ['S.No'] + [col for col in d.cols.names[:x_len]] + ['Score']
    ] + [
        ([i + 1] + r[:x_len] + [round((1 - chebyshev(d, r)) * 100, 2) if i in labelled_indices
        else "unlabelled"])
        
        for i, r in enumerate(tot_rows)
    ]

    print(rows_to_markdown(table))

    envi = RowSelectionEnv(tot_rows, labelled_indices)

