"""
ModelActivations.py

Represents the activations for one step of the model,
split across multiple layers.
"""

from typing import Dict
from src.activation_recorder.modules import LayerActivations
from src.activation_recorder.ModelInformation import ModelInformation

class ModelActivations:
    """
    For a single step, this stores a dictionary of layer_index -> LayerActivations.
    """

    def __init__(self, step_index: int, model_info: ModelInformation):
        """
        :param step_index: The generation or inference step index
        :param model_info: The ModelInformation describing the model
        """
        self.step_index = step_index
        self.model_info = model_info
        self.layers: Dict[int, LayerActivations] = {}

    def get_or_create_layer_activations(self, layer_index: int) -> LayerActivations:
        """
        Retrieve or create LayerActivations for the given layer index.
        """
        if layer_index not in self.layers:
            self.layers[layer_index] = LayerActivations(layer_index, self.model_info)
        return self.layers[layer_index]

    def get_layer_activations(self, layer_index: int) -> LayerActivations:
        """
        Return the existing LayerActivations for a layer index.
        """
        return self.layers[layer_index]

    def __len__(self):
        return len(self.layers)
    
    def is_complete(self):
        return len(self) == self.model_info.num_layers



if __name__ == "__main__":
    """
    Simple demonstration: create ModelActivations, add a couple of layers,
    and verify they're stored.
    """
    from transformers import AutoModelForCausalLM
    from activation_recorder.structures.ModelInformation import ModelInformation

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    info = ModelInformation(model)

    step_acts = ModelActivations(step_index=0, model_info=info)
    layer0 = step_acts.get_or_create_layer_activations(0)
    layer1 = step_acts.get_or_create_layer_activations(1)

    print("Created ModelActivations at step=0.")
    print("Layers stored:", list(step_acts.layers.keys()))
