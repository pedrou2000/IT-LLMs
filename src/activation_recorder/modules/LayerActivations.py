"""
LayerActivations.py

Represents the activations at a single Transformer layer for a single step.
Can contain attention, MLP, and MoE structures.
"""

from typing import Optional
from src.activation_recorder.modules.AttentionLayerActivations import AttentionLayerActivations
from src.activation_recorder.modules.MLPLayerActivations import MLPLayerActivations
from src.activation_recorder.modules.MoELayerActivations import MoELayerActivations
from src.activation_recorder.ModelInformation import ModelInformation

class LayerActivations:
    """
    For a single layer at a single step, stores sub-activations:
      - attention: AttentionLayerActivations
      - mlp: MLPLayerActivations
      - moe: MoELayerActivations
    """

    def __init__(self, layer_index: int, model_info: ModelInformation):
        """
        :param layer_index: The layer index in the model
        :param model_info: The ModelInformation describing the model
        """
        self.layer_index = layer_index
        self.model_info = model_info

        self.attention: Optional[AttentionLayerActivations] = None
        self.mlp: Optional[MLPLayerActivations] = None
        self.moe: Optional[MoELayerActivations] = None

    def get_or_create_attention(self) -> AttentionLayerActivations:
        if self.attention is None:
            self.attention = AttentionLayerActivations(self.model_info)
        return self.attention

    def get_or_create_mlp(self) -> MLPLayerActivations:
        if self.mlp is None:
            self.mlp = MLPLayerActivations(self.model_info)
        return self.mlp

    def get_or_create_moe(self) -> MoELayerActivations:
        if self.moe is None:
            self.moe = MoELayerActivations(self.model_info)
        return self.moe

    def set_attention(self, attn: AttentionLayerActivations):
        self.attention = attn

    def set_mlp(self, mlp_acts: MLPLayerActivations):
        self.mlp = mlp_acts

    def set_moe(self, moe_acts: MoELayerActivations):
        self.moe = moe_acts
    
    def __len__(self):
        return len(self.attention)
    
    def is_complete(self):
        return len(self.attention) == self.model_info.num_attention_heads_per_layer

if __name__ == "__main__":
    """
    Simple demonstration: create a LayerActivations, add attention & MLP sub-structures, and check.
    """
    from transformers import AutoModelForCausalLM
    from activation_recorder.structures.ModelInformation import ModelInformation

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    info = ModelInformation(model)

    layer_acts = LayerActivations(layer_index=0, model_info=info)
    attn = layer_acts.get_or_create_attention()
    mlp = layer_acts.get_or_create_mlp()

    print("LayerActivations for layer=0 created.")
    print("Attention object:", attn)
    print("MLP object:", mlp)
