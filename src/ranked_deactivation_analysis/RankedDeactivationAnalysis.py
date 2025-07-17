from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Union
import xarray as xr

from src.utils import template_tokenize_prompts, get_tokens_and_logits



class RankedDeactivationAnalysis:
    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: Union[List[str], List[List[str]]], chat_template: str,
        node_ranking: xr.DataArray, # dims ('source_layer', 'source_node'), values are node ranks
        max_new_tokens: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.node_ranking = node_ranking
        self.max_new_tokens = max_new_tokens

        self.tokenized_prompts = template_tokenize_prompts(
            prompts,
            tokenizer,
            prompt_template=chat_template,
            tokenize_kwargs={
                "padding": "longest",     # ← or True
                "truncation": True,
                "return_tensors": "pt",
            },
        )
        print(f"Shape of tokenized prompts: {self.tokenized_prompts.keys()}")
        print(f"Shape of tokenized prompts: {self.tokenized_prompts['all'][0].keys()}")
        print(f"Shape of tokenized prompts: {self.tokenized_prompts['all'][0]['input_ids'].shape}")
        print(f"Shape of tokenized prompts: {self.tokenized_prompts['all'][0]['attention_mask'].shape}")
    
    def run(self, deactivate_k_nodes_per_iteration: int, max_deactivated_nodes: Union[int, None] = None, micro_batch_size: int = 32):
        """
        Run the ranked deactivation analysis on the model with the given prompts.
        
        :param deactivate_k_nodes_per_iteration: Number of additional nodes to deactivate in each iteration.
        :param max_deactivated_nodes: Maximum number of nodes to deactivate in total. If None, deactivate all.
        """
        non_deactivated_tokens, non_deactivated_logits = get_tokens_and_logits(
            self.model, self.tokenized_prompts, max_new_tokens=self.max_new_tokens, micro_batch_size=micro_batch_size
        )


        return non_deactivated_logits, non_deactivated_tokens
        