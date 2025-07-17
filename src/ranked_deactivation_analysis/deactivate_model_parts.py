from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict, Union


def make_attention_ablation_hook(heads_to_ablate: list[int], head_dim: int):
    def ablation_hook(module, input, output):
        output = output.clone()
        for head in heads_to_ablate:
            start = head * head_dim
            end = (head + 1) * head_dim
            output[..., start:end] = 0.0
        return output
    return ablation_hook


def deactivate_model_parts(
    model: PreTrainedModel,
    nodes_to_deactivate: List[Tuple[int, int]],
    module_name: str = "self_attn", # "self_attn", "mlp", ...
):
    """
    Deactivate specific nodes in the model by setting their weights to zero.
    
    :param model: The pre-trained model to modify.
    :param nodes_to_deactivate: List of tuples (layer_index, node_index) indicating which nodes to deactivate.
    :param module_name: The name of the module where the nodes are located (e.g., "self_attn", "mlp").
    """

    for layer_index, node_index in nodes_to_deactivate:
        # Construct the parameter name based on the module and indices
        param_name = f"{module_name}.layers.{layer_index}.nodes.{node_index}.weight"
        
        # Create a hook to zero out the weights of the specified nodes
        hook = make_attention_ablation_hook([node_index], model.config.hidden_size // model.config.num_attention_heads)
        # Register the hook to the specified module
        module = getattr(model, module_name)
        if hasattr(module, 'register_forward_hook'):
            module.register_forward_hook(hook)
        else:
            raise ValueError(f"Module {module_name} does not support forward hooks.")
        
    print(f"Deactivated nodes: {nodes_to_deactivate} in module: {module_name}")
    return model