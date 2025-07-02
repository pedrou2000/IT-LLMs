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
        if self._includes_prompt_activations(activations):
            print('Removing prompt tokens from activations')
            activations = self._remove_prompt_activations(activations)            
        
        # Create head activations
        for head_idx in range(self.model_info.num_attention_heads_per_layer):
            head_activations = self._create_head_activations(activations, head_idx)
            print(f"Attention head {head_idx} activations captured: {head_activations}")
            attn.add_head_activations(head_activations)
        
        

    
    def _includes_prompt_activations(self, activations):
        return activations['attention_outputs'].shape[-2] > 1
    
    def _remove_prompt_activations(self, activations):
        """
        query shape: torch.Size([1, 8, 9, 256])
        attention_weights shape: torch.Size([1, 8, 9, 18])
        attention_outputs shape: torch.Size([1, 9, 8, 256])
        projected_outputs shape: torch.Size([1, 9, 8, 2304])
        We want to remove the prompt tokens from the activations, in this example, the first 8 tokens, we want the tokens at pos 9 (-1).
        """
        return {
            'queries': activations['queries'][:, :, -1:, :],
            'attention_weights': activations['attention_weights'][:, :, -1:, :],
            'attention_outputs': activations['attention_outputs'][:, :, -1:, :],
            'projected_outputs': activations['projected_outputs'][:, :, -1:, :],
        }
    
    def _create_head_activations(self, activations, head_idx):
        return AttentionHeadActivations(
            query=activations['queries'][:, head_idx, :, :].squeeze(), 
            attention_weights=activations['attention_weights'][:, head_idx, :, :].squeeze(),
            attention_outputs=activations['attention_outputs'][:, head_idx, :, :].squeeze(),
            projected_outputs=activations['projected_outputs'][:, head_idx, :, :].squeeze(),
            model_info=self.model_info
        )

    def _mlp_hook_fn(self, module, module_input, module_output):
        """
        Hook function capturing MLP submodule outputs. We create an MLPLayerActivations
        and attach it to the correct place in the bottom-up structure.
        """


        if self._current_prompt_id is None:
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

        for key, value in activations.items():
            if hasattr(value, 'shape'):
                print(f'{key} shape: {value.shape}')
        print('---')
        
        assert layer_idx == activations['layer_idx'], f'Layer index mismatch: {layer_idx} != {activations["layer_idx"]}'

        # Walk up the chain
        prompt_acts = self.multi_prompt_acts.get_or_create_prompt_activations(self._current_prompt_id, self._current_prompt_text)
        model_acts = prompt_acts.get_or_create_step_activations(self._current_step_index)
        layer_acts = model_acts.get_or_create_layer_activations(layer_idx)
        moe_layer = layer_acts.get_or_create_moe()
        
        # Remove prompt tokens from activations if present
        # if self._includes_prompt_activations(activations):
        #     print('Removing prompt tokens from activations')
        #     activations = self._remove_prompt_activations(activations)            
        
        # Create head activations
        for moe_idx in range(self.model_info.num_experts_per_tok):
            # head_activations = self._create_head_activations(activations, head_idx)
            expert_activations = MoEExpertActivations()
            print(f"MoE expert {moe_idx} activations captured: {expert_activations}")
            moe_layer.add_expert_activations(expert_activations)

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
    
    def verify_recorded_activations(self, activations=None, diff_q_size: bool = False):
        """
        Verify the recorded activations are correct in shape and value.
        """
        
        # Preliminary checks 
        if activations is None:
            activations = self.multi_prompt_acts
        assert activations is not None, 'MultiPromptActivations object is not initialized'
        assert self.prompts is not None, 'Prompts are not initialized'
        assert self.max_new_tokens is not None, 'Max new tokens are not initialized'
        assert self.tokenizer is not None, 'Tokenizer is not initialized'
        
        # Check shape of activations
        assert len(activations) == len(self.prompts), f'Expected {len(self.prompts)} prompts, got {len(activations)}'
        assert len(activations.prompts[0].steps) == self.max_new_tokens, f'Expected {self.max_new_tokens} steps, got {len(activations.prompts[0].steps)}'
        assert len(activations.prompts[0].steps[0].layers) == activations.model_info.num_layers, f'Expected {activations.model_info.num_layers} layers, got {len(activations.prompts[0].steps[0].layers)}'
        assert len(activations.prompts[0].steps[0].layers[0].attention.heads) == activations.model_info.num_attention_heads_per_layer, f'Expected {activations.model_info.num_attention_heads_per_layer} heads, got {len(activations.prompts[0].steps[0].layers[0].attention.heads)}'
        
        # Check shape of a particular head activation
        max_len_prompt = 0
        for prompt_id in activations.prompts.keys():
            prompt_activations = activations.prompts[prompt_id]
            prompt_len = len(self.tokenizer.encode(prompt_activations.prompt_text))
            print(f'prompt_len: {prompt_len}')
            max_len_prompt = max(max_len_prompt, prompt_len)
            for step_id in prompt_activations.steps.keys():
                model_activations = prompt_activations.steps[step_id]
                for layer_id in model_activations.layers.keys():
                    layer = model_activations.layers[layer_id]
                    for head_id, head in enumerate(layer.attention.heads):
                        if not diff_q_size:
                            assert head.query.shape == (activations.model_info.head_dim,), f'Expected query shape ({activations.model_info.head_dim},), got {head.query.shape}'
                        # assert head.attention_weights.shape == (max_len_prompt + self.max_new_tokens - 1,), f'Expected attention_weights shape ({max_len_prompt + self.max_new_tokens - 1},), got {head.attention_weights.shape}, for head {head_id} in layer {layer_id} at step {step_id}'
                        assert torch.all(head.attention_weights[prompt_len + step_id:] == 0), f'Expected attention_weights to be zero after prompt_len + step_id, got {head.attention_weights[prompt_len + step_id:]} for head {head_id} in layer {layer_id} at step {step_id}'
                        assert head.attention_outputs.shape == (activations.model_info.head_dim,), f'Expected attention_outputs shape ({activations.model_info.head_dim},), got {head.attention_outputs.shape} for head {head_id} in layer {layer_id} at step {step_id}'
                        assert head.projected_outputs.shape == (activations.model_info.hidden_size,), f'Expected projected_outputs shape ({activations.model_info.hidden_size},), got {head.projected_outputs.shape} for head {head_id} in layer {layer_id} at step {step_id}'

        # Print a success message
        print(f'Activations check passed!')

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
