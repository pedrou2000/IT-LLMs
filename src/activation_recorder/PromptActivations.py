"""
PromptActivations.py

Represents the activations for a single prompt across multiple steps.
"""

from typing import Dict
from src.activation_recorder.ModelInformation import ModelInformation
from src. activation_recorder.ModelActivations import ModelActivations

class PromptActivations:
    """
    For a single prompt, stores multiple steps of activations.
    Each step is represented by a ModelActivations object.
    """

    def __init__(self, prompt_id: int, prompt_text: str, model_info: ModelInformation):
        """
        :param prompt_id: An integer or unique identifier for the prompt
        :param prompt_text: The initial text for this prompt
        :param model_info: The ModelInformation describing the model
        """
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text
        self.model_info = model_info

        self.steps: Dict[int, ModelActivations] = {}
        self.prompt_completion = None

    def get_or_create_step_activations(self, step_index: int) -> ModelActivations:
        """
        Retrieve or create the ModelActivations for a particular step index.
        """
        if step_index not in self.steps:
            self.steps[step_index] = ModelActivations(step_index, self.model_info)
        return self.steps[step_index]

    def get_step_activations(self, step_index: int) -> ModelActivations:
        """
        Return the existing ModelActivations for the given step index.
        """
        return self.steps[step_index]

    def set_prompt_completion(self, completion: str):
        """
        Set the current completion string for this prompt.
        """
        self.prompt_completion = completion
    
    def verify(self, diff_q_size: bool = False, prompt_len: int = 0, max_new_tokens: int = 0):
        """ Verify the recorded activations for this prompt. Checks shapes and values of activations across all steps. """
        for step_index, step_activations in self.steps.items():
            step_activations.verify(diff_q_size, prompt_len, step_index, max_new_tokens)
    
    


if __name__ == "__main__":
    """
    Simple usage example: create a PromptActivations and add a couple of steps.
    """
    from transformers import AutoModelForCausalLM
    from activation_recorder.structures.ModelInformation import ModelInformation

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    info = ModelInformation(model)

    prompt_acts = PromptActivations(
        prompt_id=0,
        prompt_text="Hello world",
        model_info=info
    )

    step0 = prompt_acts.get_or_create_step_activations(0)
    step1 = prompt_acts.get_or_create_step_activations(1)
    prompt_acts.set_prompt_completion("Completion text so far...")

    print("PromptActivations for prompt_id=0 created. Steps:", list(prompt_acts.steps.keys()))
    print("Prompt text:", prompt_acts.prompt_text)
    print("Completion:", prompt_acts.prompt_completion)
