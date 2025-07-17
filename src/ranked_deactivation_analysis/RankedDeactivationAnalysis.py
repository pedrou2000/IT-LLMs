from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Union
import xarray as xr

from src.utils import template_tokenize_prompts, get_tokens_and_probs, invert_node_ranking



class RankedDeactivationAnalysis:
    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: Union[List[str], List[List[str]]], chat_template: str,
        node_ranking: xr.DataArray, # dims ('source_layer', 'source_node'), values are node ranks
        max_new_tokens: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.node_ranking = invert_node_ranking(node_ranking)
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


        
    
    def run(self, deactivate_k_nodes_per_iteration: int, max_deactivated_nodes: Union[int, None] = None, micro_batch_size: int = 32):
        """
        Run the ranked deactivation analysis on the model with the given prompts.
        
        :param deactivate_k_nodes_per_iteration: Number of additional nodes to deactivate in each iteration.
        :param max_deactivated_nodes: Maximum number of nodes to deactivate in total. If None, deactivate all.
        """
        # Dict[str, List[GenResult]] where GenResult = (tokens, probs, decoded_text)
        non_ablated_token_and_logits = get_tokens_and_probs( 
            model=self.model, 
            tokenizer=self.tokenizer,
            tokenized_prompts=self.tokenized_prompts, 
            max_new_tokens=self.max_new_tokens, 
            micro_batch_size=micro_batch_size
        )


        # Iterate over the node ranking and deactivate nodes
        max_deactivated_nodes = min(len(self.node_ranking), max_deactivated_nodes + 1) if max_deactivated_nodes is not None else len(self.node_ranking)
        for last_deactivated_node in range(0, max_deactivated_nodes, deactivate_k_nodes_per_iteration):
            nodes_to_deactivate = self.node_ranking[:last_deactivated_node]
            print(f"Deactivating nodes: {nodes_to_deactivate}")

            


        return non_ablated_token_and_logits
        