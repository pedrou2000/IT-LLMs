"""
ActivationRecorder.py

Implements an ActivationRecorder class that:
  - Accepts a HuggingFace Transformer model + tokenizer
  - Creates ModelInformation
  - Registers forward hooks to record attention, MLP, and MoE activations
  - Builds up a bottom-up data structure: MultiPromptActivations -> PromptActivations -> ...
  - Demonstrates usage in the __main__ block for testing
"""
import pickle, os
import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from IPython.core.debugger import Pdb

from src.activation_recorder.ModelInformation import ModelInformation
from src.activation_recorder.MultiPromptActivations import MultiPromptActivations

# Sub-structures we'll fill from hooks
from src.activation_recorder.modules import AttentionHeadActivations
from src.activation_recorder.modules import MLPLayerActivations
from src.activation_recorder.modules import MoEExpertActivations


class ActivationRecorder:
    """
    Demonstration of bottom-up recording approach.
    Each forward hook captures the lowest-level submodule activation (e.g. a single head).
    We build upward from these sub-activations into the final MultiPromptActivations container.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        :param model: A loaded Hugging Face model
        :param tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

        # Build a ModelInformation from the loaded model
        self.model_info = ModelInformation(model)
        print(f'Recorder initialized for model: {self.model_info}')

        # This container will hold everything in a bottom-up fashion
        self.multi_prompt_acts = MultiPromptActivations(self.model_info)

        # Track current prompt/step in generation
        self._current_prompt_id: Optional[int] = None
        self._current_prompt_text: Optional[str] = None
        self._current_step_index: int = 0

        # Keep references to the hooks
        self._hooks = []

    def attach_hooks(self):
        """
        Attach forward hooks to only high-level submodules of the model.
        Specifically, we attach hooks to:
        - self-attention layers (self_attn)
        - MLP layers (mlp)
        - MoE layers (if applicable)
        We avoid submodules like q_proj, k_proj, v_proj, etc.
        """
        for name, module in self.model.named_modules():
            # Only attach to self-attn and MLP at the layer level
            if name.endswith("self_attn"):  # High-level attention module
                h = module.register_forward_hook(self._attention_hook_fn)
                self._hooks.append(h)
            # elif name.endswith("mlp"):  # High-level MLP module
            #     h = module.register_forward_hook(self._mlp_hook_fn)
            #     self._hooks.append(h)
            elif name.endswith("mlp"):  # If MoE exists in this model
                h = module.register_forward_hook(self._moe_hook_fn)
                self._hooks.append(h)
        # 2) A single hook on the entire model to keep track of the step index
        h_model = self.model.register_forward_hook(self._model_forward_hook)
        self._hooks.append(h_model)

    def remove_hooks(self):
        """Detach all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def record_prompts(self, prompts: List[str], max_new_tokens: int = 20, use_cache: bool = True) -> MultiPromptActivations:
        """
        Runs autoregressive generation for each prompt and collects intermediate activations
        via forward hooks. Each new sub-activation is attached bottom-up to the final structure.
        """
        # Record the prompts and max_new_tokens
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

        # Attach hooks before generation
        self.attach_hooks()

        for prompt_id, prompt_text in enumerate(prompts):
            print(f'\n\n\n\n\nWorking on prompt {prompt_id}: {prompt_text}\n\n\n\n\n')
            self._current_prompt_id = prompt_id
            self._current_prompt_text = prompt_text
            self._current_step_index = 0 # This will be incremented inside the hooks

            # Tokenize
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            # Force cache reset (important)
            inputs["past_key_values"] = None  # Explicitly reset cache
            self.model._past = None  # Reset KV-cache
            torch.cuda.empty_cache()  # Optional: Free GPU memory
            
            # Single call to .generate(...) with caching
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,  # Enable caching
                    return_dict_in_generate=True,
                    output_attentions=False
                )

            # We only have the final completion text. Each incremental step is done internally.
            completion_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            # Store the final completion
            prompt_acts = self.multi_prompt_acts.get_or_create_prompt_activations(prompt_id, prompt_text)
            prompt_acts.set_prompt_completion(completion_text)

            print(f"\n\n\nPrompt {prompt_id} completed: {completion_text}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        # Remove hooks and return
        self.remove_hooks()
        return self.multi_prompt_acts
    
    def _model_forward_hook(self, module, module_input, module_output):
        """
        Called once per model(...) call. This is a good place to increment the
        step index, because each forward pass typically corresponds to generating
        one token (when use_cache=True).
        """
        # If we haven't started recording (prompt_id is None), skip
        if self._current_prompt_id is not None:
            print(f"** Done forward pass for step {self._current_step_index} **")
            self._current_step_index += 1
    
    def _attention_hook_fn(self, module, module_input, module_output):
        """
        Hook function capturing attention submodule outputs. We create an AttentionHeadActivations
        and attach it to the correct place in the bottom-up structure.
        """
        layer_idx = self._extract_layer_index(module)
        print(f'Hook for {module.__class__.__name__} layer {layer_idx} captured the following module output: {len(module_output)}')
        if self._current_prompt_id is None:
            return

        _, activations, _ = module_output

        for key, value in activations.items():
            if hasattr(value, 'shape'):
                print(f'{key} shape: {value.shape}')
        print('---')
        
        assert layer_idx == activations['layer_idx'], f'Layer index mismatch: {layer_idx} != {activations["layer_idx"]}'

        # Walk up the chain
        prompt_acts = self.multi_prompt_acts.get_or_create_prompt_activations(self._current_prompt_id, self._current_prompt_text)
        model_acts = prompt_acts.get_or_create_step_activations(self._current_step_index)
        layer_acts = model_acts.get_or_create_layer_activations(layer_idx)
        attn = layer_acts.get_or_create_attention()
        
        # Remove prompt tokens from activations if present
        if self._includes_prompt_activations(activations, dim=-2): # The token dimension is -2
            print('Removing prompt tokens from activations')
            activations = self._remove_prompt_activations(activations, dim=-2) # The token dimension is -2           
        
        # Create head activations
        for head_idx in range(self.model_info.num_attention_heads_per_layer):
            head_activations = self._create_head_activations(activations, layer_idx, head_idx)
            attn.add_head_activations(head_activations)
        print(f"Attention layer {layer_idx} activations captured with {len(attn.heads)} heads.")        
        

    
    def _includes_prompt_activations(self, activations, dim=-2):
        for key, value in activations.items():
            if hasattr(value, 'shape') and value.shape[dim] > 1:
                # If the last dimension is greater than 1, we assume it includes prompt activations
                return True
        return False

    def _slice_tensor_along_dim(self, tensor: torch.Tensor, dim: int = -2) -> torch.Tensor:
        """
        Return a view that keeps only the last element along `dim`
        (dimension is retained with size 1).  Silently returns the
        original object if `tensor` is not a Tensor or `dim` is out of range.
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor

        if dim >= tensor.ndim or dim < -tensor.ndim:
            return tensor                        # or raise ValueError

        dim = dim % tensor.ndim                 # normalise negative axes
        index = [slice(None)] * tensor.ndim
        index[dim] = slice(-1, None)            # keep axis, size 1
        return tensor[tuple(index)]


    
    def _remove_prompt_activations(self, activations, dim=-2):
        """
        query shape: torch.Size([1, 8, 9, 256])
        attention_weights shape: torch.Size([1, 8, 9, 18])
        attention_outputs shape: torch.Size([1, 9, 8, 256])
        projected_outputs shape: torch.Size([1, 9, 8, 2304])
        We want to remove the prompt tokens from the activations, in this example, the first 8 tokens, we want the tokens at pos 9 (-1).
        """
        for key, value in activations.items():
            activations[key] = self._slice_tensor_along_dim(value, dim=dim)
        return activations


    
    def _create_head_activations(self, activations, layer_idx, head_idx):
        return AttentionHeadActivations(
            query=activations['queries'][:, head_idx, :, :].squeeze().cpu(),
            attention_weights=activations['attention_weights'][:, head_idx, :, :].squeeze().cpu(),
            attention_outputs=activations['attention_outputs'][:, head_idx, :, :].squeeze().cpu(),
            projected_outputs=activations['projected_outputs'][:, head_idx, :, :].squeeze().cpu(),
            model_info=self.model_info,
            layer_index=layer_idx,
            head_index=head_idx
        )

    def _mlp_hook_fn(self, module, module_input, module_output):
        """
        Hook function capturing MLP submodule outputs. We create an MLPLayerActivations
        and attach it to the correct place in the bottom-up structure.
        """
        is_moe_layer = hasattr(module, 'experts')
        layer_idx = self._extract_layer_index(module)
        print(f'Hook for {module.__class__.__name__} layer {layer_idx} which is_moe_layer={is_moe_layer}, captured the following module output: {len(module_output)}')
        if self._current_prompt_id is None or is_moe_layer:
            return

        mlp_acts = MLPLayerActivations(self.model_info)
        # Fill with real or dummy data
        mlp_acts.x_prime = torch.zeros(10)
        mlp_acts.y = torch.zeros(10)

        prompt_acts = self.multi_prompt_acts.get_or_create_prompt_activations(
            self._current_prompt_id, self._current_prompt_text
        )
        step_acts = prompt_acts.get_or_create_step_activations(self._current_step_index)
        layer_idx = self._extract_layer_index(module)
        layer_acts = step_acts.get_or_create_layer_activations(layer_idx)
        layer_acts.set_mlp(mlp_acts)

    def _moe_hook_fn(self, module, module_input, module_output):
        """
        Hook function capturing MoE submodule outputs. We create a MoEExpertActivations
        and attach it to the correct place in the bottom-up structure.
        """
        is_moe_layer = hasattr(module, 'experts')
        layer_idx = self._extract_layer_index(module)
        print(f'Hook for {module.__class__.__name__} layer {layer_idx} which is_moe_layer={is_moe_layer}, captured the following module output: {len(module_output)}')
        if self._current_prompt_id is None or not is_moe_layer:
            return

        _, activations = module_output
        
        assert layer_idx == activations['layer_idx'], f'Layer index mismatch: {layer_idx} != {activations["layer_idx"]}'

        # Walk up the chain
        prompt_acts = self.multi_prompt_acts.get_or_create_prompt_activations(self._current_prompt_id, self._current_prompt_text)
        model_acts = prompt_acts.get_or_create_step_activations(self._current_step_index)
        layer_acts = model_acts.get_or_create_layer_activations(layer_idx)
        moe_layer = layer_acts.get_or_create_moe()
        
        for key, value in activations.items():
            if hasattr(value, 'shape'):
                print(f'{key} shape: {value.shape}')
        
        if 'topk_ids' in activations and 'topk_weight' in activations:
            print(f'topk_ids: {activations["topk_ids"]}')
            print(f'topk_weight: {activations["topk_weight"]}')
        print('---')     

        # Remove prompt tokens from activations if present
        if self._includes_prompt_activations_moe(activations):
            print('Removing prompt tokens from activations')
            activations = self._remove_prompt_activations_moe(activations) 
        else:
            # Squeeze the activations to remove the extra dimension
            for key, value in activations.items():
                if hasattr(value, 'shape') and len(value.shape) > 1:
                    activations[key] = value.squeeze()

        for key, value in activations.items():
            if hasattr(value, 'shape'):
                print(f'{key} shape: {value.shape}')
        
        if 'topk_ids' in activations and 'topk_weight' in activations:
            print(f'topk_ids: {activations["topk_ids"]}')
            print(f'topk_weight: {activations["topk_weight"]}')
        print('---')     
        
        self._create_and_add_moe_expert_activations(activations, layer_idx, moe_layer)
        print(f"MoE layer {layer_idx} activations captured with {len(moe_layer.experts)} experts.")
    

    
    def _includes_prompt_activations_moe(self, activations):
        topk_ids = activations.get('topk_ids', None)
        if topk_ids is None or len(topk_ids.shape) < 2:
            return False
        return activations['topk_ids'].shape[-2] > 1

    def _remove_prompt_activations_moe(self, activations):
        """
        We want to remove the prompt tokens from the activations, in this example, the first 8 tokens, we want the tokens at pos 9 (-1).
        """
        activations['topk_ids'] = self._slice_tensor_along_dim(activations['topk_ids'], dim=-2).squeeze()
        activations['topk_weights'] = self._slice_tensor_along_dim(activations['topk_weights'], dim=-2).squeeze()
        activations['out_before_mul'] = self._slice_tensor_along_dim(activations['out_before_mul'], dim=-3).squeeze()
        activations['out_after_mul'] = self._slice_tensor_along_dim(activations['out_after_mul'], dim=-3).squeeze()
        activations['shared_experts_out'] = self._slice_tensor_along_dim(activations['shared_experts_out'], dim=-2).squeeze()
        return activations

    def _create_and_add_moe_expert_activations(self, activations, layer_idx, moe_layer):
        """ Create and add MoE expert activations to the MoE layer. """

        # Create the routed experts 
        hidden_size = self.model_info.hidden_size
        topk_ids = activations['topk_ids'].squeeze() 
        topk_weights = activations['topk_weights'].squeeze()
        assert topk_ids.shape[0] == self.model_info.num_experts_per_tok, f'Expected {self.model_info.num_experts_per_tok} experts, got {topk_ids.shape}'
        assert topk_weights.shape[0] == self.model_info.num_experts_per_tok, f'Expected {self.model_info.num_experts_per_tok} experts, got {topk_weights.shape}'
        assert topk_ids.shape == topk_weights.shape == (self.model_info.num_experts_per_tok,), f'Expected topk_ids and topk_weights to have shape ({self.model_info.num_experts_per_tok},), got {topk_ids.shape} and {topk_weights.shape}'
        # topk_ids = topk_ids.tolist()  # Convert to list for easier indexing
        # topk_weights = topk_weights.tolist()  # Convert to list for easier indexing
        print(f"Top-k experts IDs: {topk_ids}, Top-k weights: {topk_weights}, creating {self.model_info.n_routed_experts} experts.")

        for expert_index in range(self.model_info.n_routed_experts):
            if expert_index in topk_ids:
                # Get the index of the expert in the topk_ids: just find the index of the expert_index in topk_ids
                expert_idx_in_weights = (topk_ids == expert_index).nonzero(as_tuple=True)[0].item()
                gate_value = torch.tensor(topk_weights[expert_idx_in_weights])
                mlp_output = activations['out_before_mul'][expert_idx_in_weights, :].squeeze()
                expert_output = activations['out_after_mul'][expert_idx_in_weights, :].squeeze()
                expert_activations = MoEExpertActivations(
                    gate_value=gate_value.cpu(),  
                    mlp_output=mlp_output.cpu(),
                    expert_output=expert_output.cpu(),
                    model_info=self.model_info,
                    layer_index=layer_idx,
                    expert_index=expert_index
                )

            else:
                # If the expert is not in the topk_ids, we set gate_value to 0 and outputs to zero tensors
                expert_activations = MoEExpertActivations.empty(
                    layer_index=layer_idx,
                    expert_index=expert_index,
                    is_shared=False,  # This is not a shared expert
                    model_info=self.model_info
                )
            moe_layer.add_expert_activations(expert_activations)
            
        
        # Create the shared expert activations
        expert_index += 1
        shared_expert = MoEExpertActivations(
            gate_value = torch.tensor(1.0),  # Shared expert always has gate value 1.0 
            mlp_output = activations['shared_experts_out'].squeeze(),
            expert_output = activations['shared_experts_out'].squeeze(),
            is_shared = True,
            model_info = self.model_info,
            layer_index = layer_idx,
            expert_index = expert_index # Shared expert index is always the last one
        )
        print(f"Adding expert {expert_index} activations: {repr(expert_activations)}")
        moe_layer.add_expert_activations(shared_expert)




    def _extract_layer_index(self, module) -> int:
        """
        Simple utility to parse the layer index from the module name. Adjust to your architecture.
        """
        full_name = ""
        for nm, mod in self.model.named_modules():
            if mod is module:
                full_name = nm
                break

        tokens = full_name.split(".")
        for i, t in enumerate(tokens):
            if t.isdigit():
                return int(t)
            if t in ("layers", "h") and (i + 1) < len(tokens) and tokens[i + 1].isdigit():
                return int(tokens[i + 1])
        return 0
    

if __name__ == "__main__":
    """
    Simple usage of ActivationRecorder on a small model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    

    model_name = "google/gemma-2-2b-it"
    max_new_tokens=10
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        # device_map='auto', 
        attn_implementation='eager',  
    )
    model.eval()

    recorder = ActivationRecorder(model, tokenizer)
    prompts = ["Hello, world! How can you code",  "Tell me a joke"]
    activations = recorder.record_prompts(prompts, max_new_tokens=max_new_tokens)
    recorder.verify_recorded_activations(activations)
    
    # Save the activations to disk.
    save_dir = "./data/activations"
    activations.save(save_dir)
    
    # Load the activations from disk.
    file_path = os.path.join(save_dir, "multi_prompt_activations.pkl")
    loaded_activations = MultiPromptActivations.load(file_path)
    
    # Optional: verify the loaded activations match the saved ones.
    print("Loaded MultiPromptActivations object has:", len(loaded_activations.prompts), "prompts recorded.")
    
    # Check again the activations
    recorder.verify_recorded_activations(loaded_activations)
    

    print("Final MultiPromptActivations object has:", len(activations.prompts), "prompts recorded.")
    
    # Extract the first prompt, first step, first layer, first head
    prompt_acts = activations.prompts[0]
    step_acts = prompt_acts.steps[0]
    layer_acts = step_acts.layers[0]
    attn = layer_acts.attention
    for head_acts in attn.heads:
        print(head_acts.query.shape)
        print(head_acts.attention_weights.shape)
        print(head_acts.attention_outputs.shape)
        print(head_acts.projected_outputs.shape)
