import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Union, Dict
import xarray as xr
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt

from src.utils import template_tokenize_prompts, get_tokens_and_probs, invert_node_ranking, get_teacher_forcing_tokens_and_probs
from src.ranked_deactivation_analysis.deactivate_model_parts import deactivate_model_parts

@dataclass
class PerformanceDivergenceResult:
    kl_per_prompt_per_timestep: Dict[str, List[torch.Tensor]]  # [prompt_key][prompt_idx] -> tensor of shape (T,)
    performance_divergence_per_prompt: Dict[str, List[float]]  # [prompt_key][prompt_idx] -> float (average KL)
    overall_performance_divergence: float  # Average across all prompts
    num_nodes_deactivated: int
    deactivated_nodes: List[Tuple[int, int]]  # List of (layer, node) tuples

@dataclass
class RankedDeactivationResults:
    non_deactivated_results: Dict[str, List[tuple]]  # GenResult = (tokens, probs, decoded_text)
    deactivation_results: List[PerformanceDivergenceResult]  # One per iteration
    deactivation_schedule: List[int]  # Number of nodes deactivated at each iteration

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
        # Randomly shuffle the node ranking list several times to ensure randomness
        # random.shuffle(self.node_ranking)

        self.max_new_tokens = max_new_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  

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
    
    def compute_kl_divergence(
        self,
        non_deactivated_results: Dict[str, List[tuple]],
        deactivated_results: Dict[str, List[tuple]],
        num_nodes_deactivated: int,
        deactivated_nodes: List[Tuple[int, int]]
    ) -> PerformanceDivergenceResult:
        """
        Compute KL divergence between non-deactivated and deactivated model results.
        """
        kl_per_prompt_per_timestep = {}
        performance_divergence_per_prompt = {}
        all_divergences = []
        
        for prompt_key in non_deactivated_results:
            kl_per_prompt_per_timestep[prompt_key] = []
            performance_divergence_per_prompt[prompt_key] = []
            
            for i, (non_deact_result, deact_result) in enumerate(zip(
                non_deactivated_results[prompt_key], 
                deactivated_results[prompt_key]
            )):
                # Extract probability tensors
                non_deact_probs = non_deact_result[1]  # (T, |V|)
                deact_probs = deact_result[1]         # (T, |V|)
                print(f"Processing prompt {prompt_key}, iteration {i}:")
                print(f"Non-deactivated probs shape: {non_deact_probs.shape}, Deactivated probs shape: {deact_probs.shape}")
                # print the sum of differences in values for debugging
                # print(f"Sum of differences: {torch.sum(torch.abs(non_deact_probs - deact_probs)):.6f}")
                
                # Ensure tensors are on the same device and have same shape
                assert non_deact_probs.shape == deact_probs.shape, f"Shape mismatch: {non_deact_probs.shape} vs {deact_probs.shape}"
                
                # Compute KL divergence at each timestep: KL(p_non_deact || p_deact)
                # KL(P||Q) = sum(P * log(P/Q))
                eps = 1e-30
                p = non_deact_probs.float().clamp_min(eps)
                q = deact_probs.float().clamp_min(eps)
                kl_per_timestep = torch.sum(p * torch.log(p / q), dim=-1)
                print(f"Prompt {prompt_key}, Iteration {i}: KL divergence per timestep: {kl_per_timestep}")
                
                # Store per-timestep KL
                kl_per_prompt_per_timestep[prompt_key].append(kl_per_timestep)
                
                # Compute performance divergence for this prompt (average over timesteps)
                perf_div = torch.mean(kl_per_timestep).item()
                performance_divergence_per_prompt[prompt_key].append(perf_div)
                all_divergences.append(perf_div)
        
        # Overall performance divergence
        overall_perf_div = sum(all_divergences) / len(all_divergences) if all_divergences else 0.0
        
        return PerformanceDivergenceResult(
            kl_per_prompt_per_timestep=kl_per_prompt_per_timestep,
            performance_divergence_per_prompt=performance_divergence_per_prompt,
            overall_performance_divergence=overall_perf_div,
            num_nodes_deactivated=num_nodes_deactivated,
            deactivated_nodes=deactivated_nodes
        )
    
    def run(self, deactivate_k_nodes_per_iteration: int, max_deactivated_nodes: Union[int, None] = None, micro_batch_size: int = 32) -> RankedDeactivationResults:
        """
        Run the ranked deactivation analysis on the model with the given prompts.
        
        :param deactivate_k_nodes_per_iteration: Number of additional nodes to deactivate in each iteration.
        :param max_deactivated_nodes: Maximum number of nodes to deactivate in total. If None, deactivate all.
        """
        print("Getting non-deactivated model results...")
        # Dict[str, List[GenResult]] where GenResult = (tokens, probs, decoded_text)
        non_deactivated_token_and_logits = get_tokens_and_probs( 
            model=self.model, 
            tokenizer=self.tokenizer,
            tokenized_prompts=self.tokenized_prompts, 
            max_new_tokens=self.max_new_tokens, 
            micro_batch_size=micro_batch_size
        )
        div_self = self.compute_kl_divergence(non_deactivated_token_and_logits,
                                          non_deactivated_token_and_logits,
                                          num_nodes_deactivated=0,
                                          deactivated_nodes=[])
        print(f"Self-KL divergence (baseline): {div_self.overall_performance_divergence:.6f}")

        deactivation_results = []
        deactivation_schedule = []

        # Iterate over the node ranking and deactivate nodes
        total_nodes = len(self.node_ranking)
        max_deactivated_nodes = min(total_nodes, max_deactivated_nodes) if max_deactivated_nodes is not None else total_nodes
        
        print(f"Starting deactivation analysis: {max_deactivated_nodes} max nodes, {deactivate_k_nodes_per_iteration} per iteration")
        
        for iteration, last_deactivated_node in enumerate(range(0, max_deactivated_nodes + 1, deactivate_k_nodes_per_iteration)):
            nodes_to_deactivate = self.node_ranking[:last_deactivated_node]
            num_nodes_to_deactivate = len(nodes_to_deactivate)
            deactivated_nodes_list = [(int(layer), int(node)) for layer, node in nodes_to_deactivate]
            
            print(f"\nIteration {iteration + 1}: Deactivating {num_nodes_to_deactivate} nodes")
            print(f"Sample deactivated nodes: {deactivated_nodes_list[:5]}{'...' if len(deactivated_nodes_list) > 5 else ''}")

            # Deactivate the nodes in the model temporarily only for this iteration
            with deactivate_model_parts(
                model=self.model,
                nodes_to_deactivate=nodes_to_deactivate,
                module_name="self_attn"  # or "mlp", etc.
            ) as deactivated_model:
                # Re-run the generation with the deactivated nodes
                # Dict[str, List[GenResult]] where GenResult = (tokens, probs, decoded_text)
                print(f"Getting deactivated model results for {num_nodes_to_deactivate} deactivated nodes...")
                deactivated_token_and_logits = get_teacher_forcing_tokens_and_probs( 
                    model=deactivated_model, 
                    tokenizer=self.tokenizer,
                    non_deactivated_token_and_logits=non_deactivated_token_and_logits,
                    micro_batch_size=micro_batch_size,
                    sample_alternative_tokens=False  # Set to True if you want to see what tokens would be generated
                )
                
                # Compute KL divergence
                print("Computing KL divergence...")
                divergence_result = self.compute_kl_divergence(
                    non_deactivated_results=non_deactivated_token_and_logits,
                    deactivated_results=deactivated_token_and_logits,
                    num_nodes_deactivated=num_nodes_to_deactivate,
                    deactivated_nodes=deactivated_nodes_list
                )
                
                print(f"Overall performance divergence: {divergence_result.overall_performance_divergence:.6f}")
                
                deactivation_results.append(divergence_result)
                deactivation_schedule.append(num_nodes_to_deactivate)
                
                # Clear GPU memory
                del deactivated_token_and_logits
                torch.cuda.empty_cache()

        return RankedDeactivationResults(
            non_deactivated_results=non_deactivated_token_and_logits,
            deactivation_results=deactivation_results,
            deactivation_schedule=deactivation_schedule
        )
    
    def plot_performance_divergence(self, results: RankedDeactivationResults):
        """
        Plot the performance divergence as a function of number of deactivated nodes.
        """
        x = results.deactivation_schedule
        y = [result.overall_performance_divergence for result in results.deactivation_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Number of Deactivated Nodes')
        plt.ylabel('Overall Performance Divergence (KL)')
        plt.title('Performance Divergence vs Number of Deactivated Nodes')
        plt.grid(True, alpha=0.3)
        plt.show()