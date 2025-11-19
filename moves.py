
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import List, Optional
import json

from langchain_core.pydantic_v1 import BaseModel, Field

def format_train_set(train_set, k=3):
    """Return up to k examples formatted cleanly."""
    subset = train_set[:k]
    formatted = ""
    for ex in subset:
        formatted += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
    return formatted


TRAIN_SAMPLE= 3

class GenerationMove:
    """Base class for prompt transformations."""
    def __init__(self, name: str, llm: ChatOpenAI = None):
        self.name = name
        self.llm = llm

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        """Apply transformation to prompt string and return new variant."""
        raise NotImplementedError


class VerboseMove(GenerationMove):
    """Expand the prompt with more detail and clarity using LLM."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("make_verbose", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        examples = format_train_set(train_set, TRAIN_SAMPLE)

        expansion_prompt = f"""You are an expert prompt engineer.

Your task is to REWRITE the following prompt to be more detailed,
thorough, explicit, and comprehensive. You may expand the explanation,
add clarifying details, break the instructions into multiple sections,
give more guidance, or otherwise make the task description richer.

BUT YOU MUST FOLLOW ONE CRITICAL RULE:

### CRITICAL RULE — DO NOT CHANGE THE REQUIRED OUTPUT FORMAT
- The rewritten prompt must still produce outputs with the SAME format,
  structure, level of detail, and overall form as shown in the examples below.
- You may expand the INSTRUCTIONS,
  but you may NOT change what the OUTPUT is supposed to look like.
- The rewritten prompt must still lead the model to generate the SAME TYPE
  of output as the examples demonstrate.

- DO NOT add any examples.
- DO NOT add sample inputs or outputs.
- DO NOT describe the task using the training examples.

=== REQUIRED OUTPUT FORMAT (from training examples) ===
{examples}


Original prompt:
{prompt}

Expanded prompt:"""
        try:
            result = self.llm.invoke([{"role": "user", "content": expansion_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in VerboseMove: {e}")
            return prompt


class ChainOfThoughtMove(GenerationMove):
    """Add chain of thought reasoning to the prompt."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("add_cot", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        cot_prompt = f"""You are an expert prompt engineer. Take the following prompt and add chain-of-thought reasoning instructions. 
Add guidance that encourages the model to break down the problem into steps, think through the reasoning, and show intermediate steps.
Make the prompt encourage step-by-step thinking and reasoning.

Original prompt:
{prompt}

Prompt with chain-of-thought:"""
        
        try:
            result = self.llm.invoke([{"role": "user", "content": cot_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in ChainOfThoughtMove: {e}")
            return prompt


class ShortenMove(GenerationMove):
    """Make the prompt more concise using LLM."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("make_concise", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        examples = format_train_set(train_set, TRAIN_SAMPLE)
        shorten_prompt = f"""You are an expert prompt engineer.

Rewrite the following prompt to be shorter, more concise, and more direct.
Reduce unnecessary wording, but keep the task fully clear and complete.

### CRITICAL RULE — DO NOT CHANGE THE REQUIRED OUTPUT FORMAT
- The rewritten prompt must still lead the model to produce outputs matching
  the SAME format, structure, and type shown in the training examples.
- You may shorten, condense, or simplify instructions,
  but the output the model produces should be unchanged in form.
- DO NOT add any examples.
- DO NOT add sample inputs or outputs.
- DO NOT describe the task using the training examples.

=== REQUIRED OUTPUT FORMAT (from training examples) ===
{examples}

Original prompt:
{prompt}

Concise prompt:"""
        
        try:
            result = self.llm.invoke([{"role": "user", "content": shorten_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in ShortenMove: {e}")
            return prompt


class ReorderMove(GenerationMove):
    """Reorder the prompt for better clarity using LLM."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("reorder", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        examples = format_train_set(train_set, TRAIN_SAMPLE)
        reorder_prompt = f"""You are an expert prompt engineer.

Rewrite the following prompt by reorganizing and reordering its content to
maximize clarity, flow, and logical structure. You may rearrange sections,
move definitions earlier or later, or group related instructions together.

### CRITICAL RULE — DO NOT CHANGE THE REQUIRED OUTPUT FORMAT
- The reordered prompt must still lead the model to produce outputs matching
  the SAME format, structure, and type shown in the training examples.
- You may reorganize the INSTRUCTIONS,
  but you may NOT alter the kind of output expected.

- DO NOT add any examples.
- DO NOT add sample inputs or outputs.
- DO NOT describe the task using the training examples.

=== REQUIRED OUTPUT FORMAT (from training examples) ===
{examples}

Original prompt:
{prompt}

Reordered prompt:"""
        
        try:
            result = self.llm.invoke([{"role": "user", "content": reorder_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in ReorderMove: {e}")
            return prompt


class AddExamplesMove(GenerationMove):
    """Add examples to the prompt using LLM."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("add_examples", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        examples = format_train_set(train_set, TRAIN_SAMPLE)
        examples_prompt = f"""You are an expert prompt engineer.

Rewrite the following prompt by adding 1–2 short input/output examples that
help clarify what the model should do. You can choose any examples you like.

### CRITICAL RULE — DO NOT CHANGE THE REQUIRED OUTPUT FORMAT
- The rewritten prompt must still lead the model to produce outputs matching
  the SAME format, structure, and type shown in the training examples.
- You may add examples to clarify the task,
  but you may NOT alter the form of the output itself.

=== REQUIRED OUTPUT FORMAT (from training examples) ===
{examples}

Original prompt:
{prompt}

Prompt with examples:"""
        
        try:
            result = self.llm.invoke([{"role": "user", "content": examples_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in AddExamplesMove: {e}")
            return prompt


class AddConstraintsMove(GenerationMove):
    """Add explicit constraints to the prompt to improve reliability."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("add_constraints", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        constraint_prompt = f"""
You are an expert prompt engineer. Take the following prompt and add explicit constraints that help the model be reliable and consistent.

Add ALL of the following constraints:
- Do not hallucinate facts.
- Use only information explicitly provided in the input.
- Keep the answer concise and focused.
- Follow a clear and consistent output format.
- If uncertain, choose the most reasonable answer supported by the input.
- Double check the reasoning before finalizing the answer.
- Avoid adding any extraneous or creative content.

Original prompt:
{prompt}

Prompt with constraints added:
"""
        try:
            result = self.llm.invoke([{"role": "user", "content": constraint_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in AddConstraintsMove: {e}")
            return prompt



class RoleAssignmentMove(GenerationMove):
    """Add a strong persona / role to guide model behavior."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("add_role", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        role_prompt = f"""
You are an expert prompt engineer. Take the following prompt and prepend a strong role assignment that clarifies the model's persona.

Add a role such as:
- "You are an expert data annotator."
- "You are a highly precise and detail-oriented analyst."
- "You are a domain-expert reasoning assistant."

Pick one role that best fits the prompt and integrate it naturally and effectively.

Original prompt:
{prompt}

Prompt with role assignment added:
"""
        try:
            result = self.llm.invoke([{"role": "user", "content": role_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in RoleAssignmentMove: {e}")
            return prompt


class AddDefinitionsMove(GenerationMove):
    """Add clear label or category definitions to make the task explicit."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("add_definitions", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        definitions_prompt = f"""
You are an expert prompt engineer. Take the following prompt and add concise definitions for the possible outputs or categories involved in the task.

For example:
- Define sentiment labels (positive, negative, neutral).
- Define entailment, contradiction, neutrality (for NLI).
- Define what counts as a correct answer in QA.
- Define what a good summary must include or avoid.

Add only the definitions that make sense for the given prompt.

Original prompt:
{prompt}

Prompt with definitions added:
"""
        try:
            result = self.llm.invoke([{"role": "user", "content": definitions_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in AddDefinitionsMove: {e}")
            return prompt


class RephraseInputOutputInstructionsMove(GenerationMove):
    """Rephrase and clarify how the input should be interpreted and the output should be formatted."""
    def __init__(self, llm: ChatOpenAI):
        super().__init__("rephrase_io_instructions", llm)

    def apply(self, prompt: str, train_set : List[dict]) -> str:
        io_prompt = f"""
You are an expert prompt engineer. Take the following prompt and rephrase the input-output instructions for maximum clarity.

Improve structure by:
- Using explicit sections like "Task:", "Input:", "Output:".
- Defining what exactly the model should extract, answer, or generate.
- Making the required output format fully explicit.
- Reordering content to follow: Task → Input → Requirements → Output Format.

Do NOT change the meaning of the original task.

Original prompt:
{prompt}

Prompt with improved input/output instructions:
"""
        try:
            result = self.llm.invoke([{"role": "user", "content": io_prompt}])
            return result.content.strip()
        except Exception as e:
            print(f"Error in RephraseInputOutputInstructionsMove: {e}")
            return prompt
