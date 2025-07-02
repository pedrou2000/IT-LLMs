"""
AttentionLayerActivations.py

Holds classes related to storing attention head activations
and an attention container for a single layer & step.
"""

import torch
from src.activation_recorder.ModelInformation import ModelInformation

class AttentionHeadActivations:
    """
    Stores raw data for a single attention head:
      - query
      - attention_weights
      - attention_outputs (before final projection)
      - projected_outputs (after projection)
    """

    def __init__(
        self,
        query: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_outputs: torch.Tensor,
        projected_outputs: torch.Tensor,
        model_info: ModelInformation = None
    ):
        self.model_info = model_info
        self.query = query
        self.attention_weights = attention_weights
        self.attention_outputs = attention_outputs
        self.projected_outputs = projected_outputs
    
    def __repr__(self):
        return (f"AttentionHeadActivations(query={self.query.shape}, "
                f"attention_weights={self.attention_weights.shape}, "
                f"attention_outputs={self.attention_outputs.shape}, "
                f"projected_outputs={self.projected_outputs.shape}, "
                f"model_info={self.model_info})")


class AttentionLayerActivations:
    """
    Container for all attention heads in a single layer at a single step.
    """

    def __init__(self, model_info: ModelInformation):
        self.model_info = model_info
        self.heads = []  # type: list[AttentionHeadActivations]

    def add_head_activations(self, head_acts: AttentionHeadActivations):
        self.heads.append(head_acts)

    def get_head_activations(self, index: int) -> AttentionHeadActivations:
        return self.heads[index]

    def __len__(self):
        return len(self.heads)
    
    def is_complete(self):
        return len(self) == self.model_info.num_attention_heads_per_layer

    def __repr__(self):
        return f"AttentionLayerActivations(num_heads={len(self.heads)}, model_info={self.model_info})"


if __name__ == "__main__":
    """
    Simple test of creating some dummy attention heads and storing them.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    from activation_recorder.structures.ModelInformation import ModelInformation

    info = ModelInformation(model)
    attn_layer = AttentionLayerActivations(info)

    head1 = AttentionHeadActivations(
        query=torch.randn(4),
        key=torch.randn(4),
        value=torch.randn(4),
        attention_weights=torch.randn(8),
        attention_outputs=torch.randn(4),
        output=torch.randn(4),
        model_info=info
    )

    attn_layer.add_head_activations(head1)
    print("Number of heads:", len(attn_layer.heads))
    print("First head's query vector:", attn_layer.get_head_activations(0).query)
