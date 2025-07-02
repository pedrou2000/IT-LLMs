"""
ModelInformation.py

Holds a ModelInformation class that extracts key metadata directly from a
Hugging Face model, including number of layers, attention heads, etc.
"""

from transformers import PreTrainedModel

class ModelInformation:
    """
    Simple container for metadata about a Hugging Face model.
    The constructor takes the loaded HF model and extracts relevant info from model.config.
    """

    def __init__(self, hf_model: PreTrainedModel):
        """
        :param hf_model: A loaded Hugging Face model (e.g., GPT-2, LLaMA, T5, etc.)
        """
        print(hf_model.config)
        
        # Try to extract known fields from model.config.
        self.model_name: str = ""
        if hasattr(hf_model.config, "architectures") and hf_model.config.architectures:
            self.model_architecture = hf_model.config.architectures[0]
        if hasattr(hf_model.config, "name_or_path"):
            self.model_name = hf_model.config.name_or_path

        # Number of transformer layers
        if hasattr(hf_model.config, "num_hidden_layers"):
            self.num_layers: int = hf_model.config.num_hidden_layers
        elif hasattr(hf_model.config, "n_layer"):
            self.num_layers: int = hf_model.config.n_layer
        else:
            self.num_layers = 0

        # Number of attention heads per layer
        if hasattr(hf_model.config, "num_attention_heads"):
            self.num_attention_heads_per_layer: int = hf_model.config.num_attention_heads
        elif hasattr(hf_model.config, "n_head"):
            self.num_attention_heads_per_layer: int = hf_model.config.n_head
        else:
            self.num_attention_heads_per_layer = 0

        # Compute total number of attention heads if relevant
        self.total_num_attention_heads: int = self.num_layers * self.num_attention_heads_per_layer
        
        # Hidden size
        if hasattr(hf_model.config, "hidden_size"):
            self.hidden_size: int = hf_model.config.hidden_size
        elif hasattr(hf_model.config, "n_embd"):
            self.hidden_size: int = hf_model.config.n_embd
        else:
            self.hidden_size = 0
        
        # Attention head size
        if hasattr(hf_model.config, "head_dim"):
            self.head_dim: int = hf_model.config.head_dim
        elif hasattr(hf_model.config, "v_head_dim"):
            self.head_dim: int = hf_model.config.v_head_dim

        # If your model config has an attribute for attention implementation
        self.attention_implementation: str = getattr(hf_model.config, "attn_implementation", "default")

    def __repr__(self):
        return (
            f"ModelInformation("
            f"model_name={self.model_name}, "
            f"model_architecture={self.model_architecture}, "
            f"num_layers={self.num_layers}, "
            f"num_attention_heads_per_layer={self.num_attention_heads_per_layer}, "
            f"total_num_attention_heads={self.total_num_attention_heads}, "
            f"attention_implementation={self.attention_implementation}, "
            f"hidden_size={self.hidden_size}, "
            f"head_dim={self.head_dim}, "
            f"attention_implementation={self.attention_implementation}"
            f")"
        )


if __name__ == "__main__":
    """
    Simple test demonstrating how to instantiate ModelInformation with a small HF model.
    """
    from transformers import AutoModelForCausalLM

    # For quick test, use a small model
    model_name = "google/gemma-2-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    info = ModelInformation(model)
    print(info)
