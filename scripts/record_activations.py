import sys
from pathlib import Path

# Get the absolute path to the directory containing the script
SCRIPT_DIR = Path(__file__).resolve().parent

# Add the parent of the script directory to sys.path
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from src.activation_recorder import ActivationRecorder, MultiPromptActivations
from IPython.core.debugger import Pdb

load_from_disk = False

model_name = "deepseek-ai/DeepSeek-V2-Lite"
save_dir = "../data/activations"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map='cuda:0', 
    attn_implementation='eager',  
    trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.eval()

if not load_from_disk:
    max_new_tokens=10
    recorder = ActivationRecorder(model, tokenizer)
    prompts = ["Tell me a joke", "Hello, world! How can you code"]
    activations = recorder.record_prompts(prompts, max_new_tokens=max_new_tokens)
    activations.verify_recorded_activations(prompts=prompts, max_new_tokens=max_new_tokens, tokenizer=tokenizer, diff_q_size=True)
    activations.save(save_dir)

# Load the activations from disk.
file_path = os.path.join(save_dir, "multi_prompt_activations.pkl")
loaded_activations = MultiPromptActivations.load(file_path)

# Optional: verify the loaded activations match the saved ones.
print("Loaded MultiPromptActivations object has:", len(loaded_activations.prompts), "prompts recorded.")

# Check again the activations
loaded_activations.verify_recorded_activations(prompts=prompts, max_new_tokens=max_new_tokens, tokenizer=tokenizer, diff_q_size=True)

print("Final MultiPromptActivations object has:", len(activations.prompts), "prompts recorded.")

# Extract the first prompt, first step, first layer, first head
prompt_acts = activations.prompts[0]
step_acts = prompt_acts.steps[0]
layer_acts = step_acts.layers[0]
attn = layer_acts.attention
for head_idx, head_acts in attn.heads.items():
    print(head_acts.query.shape)
    print(head_acts.attention_weights.shape)
    print(head_acts.attention_outputs.shape)
    print(head_acts.projected_outputs.shape)


moe_layer_acts = step_acts.layers[1].moe
for expert_acts in moe_layer_acts.experts:
    print(repr(expert_acts))