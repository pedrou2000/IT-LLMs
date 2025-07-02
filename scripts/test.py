from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)

print(type(model))
# test2
