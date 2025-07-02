"""
MoELayerActivations.py

Holds classes for storing Mixture-of-Experts (MoE) activations,
including gating info and per-expert data.
"""

import torch
from typing import Dict
from activation_recorder.structures.ModelInformation import ModelInformation

class MoEExpertActivations:
    """
    Stores data for a single expert in an MoE layer:
      - expert_index
      - x_prime
      - x_double_prime
      - beta (gating weight, etc.)
      - y (final output)
    """

    def __init__(
        self,
        expert_index: int,
        x_prime: torch.Tensor,
        x_double_prime: torch.Tensor,
        beta: float,
        y: torch.Tensor,
        model_info: ModelInformation = None
    ):
        self.model_info = model_info
        self.expert_index = expert_index
        self.x_prime = x_prime
        self.x_double_prime = x_double_prime
        self.beta = beta
        self.y = y


class MoELayerActivations:
    """
    Container for multiple experts in a single MoE layer at a single step.
    """

    def __init__(self, model_info: ModelInformation):
        self.model_info = model_info
        self.experts: Dict[int, MoEExpertActivations] = {}

    def add_expert_activations(self, expert_acts: MoEExpertActivations):
        idx = expert_acts.expert_index
        self.experts[idx] = expert_acts

    def get_expert_activations(self, expert_index: int) -> MoEExpertActivations:
        return self.experts[expert_index]


if __name__ == "__main__":
    """
    Simple example: create a MoELayerActivations, add an expert, and check data.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    from activation_recorder.structures.ModelInformation import ModelInformation
    info = ModelInformation(model)

    moe_layer = MoELayerActivations(info)
    expert_acts = MoEExpertActivations(
        expert_index=0,
        x_prime=torch.randn(3),
        x_double_prime=torch.randn(3),
        beta=0.75,
        y=torch.randn(3),
        model_info=info
    )
    moe_layer.add_expert_activations(expert_acts)

    print("MoE layer has experts:", list(moe_layer.experts.keys()))
    print("Expert 0 x_prime:", moe_layer.get_expert_activations(0).x_prime)
