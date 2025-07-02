"""
MultiPromptActivations.py

Holds the top-level container for storing activations across multiple prompts.
Each prompt has its own PromptActivations object.
"""

from __future__ import annotations
import os, pickle
from typing import Dict
from activation_recorder.structures.PromptActivations import PromptActivations
from activation_recorder.structures.ModelInformation import ModelInformation

class MultiPromptActivations:
    """
    Top-level container for all prompt-based activations, keyed by prompt_id.
    """

    def __init__(self, model_info: ModelInformation):
        """
        :param model_info: The ModelInformation object describing the model.
        """
        self.model_info = model_info
        self.prompts: Dict[int, PromptActivations] = {}

    def get_or_create_prompt_activations(self, prompt_id: int, prompt_text: str) -> PromptActivations:
        """
        Retrieves or creates a PromptActivations object for the given prompt_id.
        """
        if prompt_id not in self.prompts:
            self.prompts[prompt_id] = PromptActivations(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                model_info=self.model_info
            )
        return self.prompts[prompt_id]

    def get_prompt_activations(self, prompt_id: int) -> PromptActivations:
        """
        Return the PromptActivations for a given prompt_id (if already created).
        """
        return self.prompts[prompt_id] 
    
    def __len__(self):
        """
        Return the number of prompts recorded.
        """
        return len(self.prompts)

    def save(self, dir_path: str) -> None:
        """
        Save the MultiPromptActivations object to a pickle file within the specified directory.
        If the directory does not exist, it will be created.

        :param dir_path: Directory where the pickle file will be saved.
        """
        try:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, "multi_prompt_activations.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            print(f"MultiPromptActivations successfully saved to '{file_path}'.")
        except Exception as e:
            print(f"Error while saving MultiPromptActivations to '{dir_path}': {e}")
            raise

    @classmethod
    def load(cls, file_path: str) -> MultiPromptActivations:
        """
        Load and return a MultiPromptActivations object from the specified pickle file.

        :param file_path: Full path to the pickle file containing the saved activations.
        :return: The loaded MultiPromptActivations object.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at '{file_path}'.")
        try:
            with open(file_path, "rb") as f:
                loaded_obj = pickle.load(f)
            if not isinstance(loaded_obj, cls):
                raise TypeError(f"Loaded object is not a MultiPromptActivations instance. Got type: {type(loaded_obj)}")
            print(f"MultiPromptActivations successfully loaded from '{file_path}'.")
            return loaded_obj
        except Exception as e:
            print(f"Error while loading MultiPromptActivations from '{file_path}': {e}")
            raise

if __name__ == "__main__":
    """
    Simple test of MultiPromptActivations creation and usage.
    """
    from activation_recorder.structures.ModelInformation import ModelInformation
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model_info = ModelInformation(model)

    mpa = MultiPromptActivations(model_info)

    # Create a couple of prompts
    prompt_acts = mpa.get_or_create_prompt_activations(0, "Hello world")
    prompt_acts.set_prompt_completion("Hello world completion")

    print("Stored prompts so far:", list(mpa.prompts.keys()))
    print("Prompt 0 text:", mpa.get_prompt_activations(0).prompt_text)
    print("Prompt 0 completion:", mpa.get_prompt_activations(0).prompt_completion)

    # Test saving the activations.
    save_dir = "./data/activations"
    mpa.save(save_dir)

    # Test loading the activations.
    file_path = os.path.join(save_dir, "multi_prompt_activations.pkl")
    loaded_mpa = MultiPromptActivations.load(file_path)

    # Verify that the loaded data matches the original.
    print("Loaded prompts:", list(loaded_mpa.prompts.keys()))
    print("Loaded Prompt 0 text:", loaded_mpa.get_prompt_activations(0).prompt_text)
    print("Loaded Prompt 0 completion:", loaded_mpa.get_prompt_activations(0).prompt_completion)