from typing import Union, List, Dict, Tuple, Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding
from torch.nn.utils.rnn import pad_sequence

def apply_prompt_template(
    prompt: str, 
    tokenizer: PreTrainedTokenizer,
    prompt_template: str
) -> str:
    if prompt_template == 'chat':
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif prompt_template == 'base':
        return f"Question: {prompt}\n Answer: "
    elif prompt_template == 'no':
        return prompt
    else:
        raise ValueError("Invalid prompt template type. Choose 'chat', 'base', or 'none'.")

PromptLike = Dict[str, List[str]]
TokenisedPrompt = Dict[str, List[BatchEncoding]]

def template_tokenize_single_prompt(prompt: str, tokenizer: PreTrainedTokenizer, *, prompt_template: str = "no", tokenize_kwargs: Dict = None) -> BatchEncoding:
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    templated = apply_prompt_template(prompt, tokenizer, prompt_template)
    return tokenizer(templated, **tokenize_kwargs)

def template_tokenize_prompts(prompts: PromptLike, tokenizer: PreTrainedTokenizer, *, prompt_template: str = "no", tokenize_kwargs: Dict = None) -> TokenisedPrompt:
    return {k: [template_tokenize_single_prompt(p, tokenizer, prompt_template=prompt_template, tokenize_kwargs=tokenize_kwargs) for p in v] for k, v in prompts.items()}

GenResult = Tuple[
    torch.LongTensor,       # generated token ids,  (P+T,) where P is prompt length and T is generated length
    torch.FloatTensor,      # probs over vocab,     (T , |V|)
    str                     # decoded text
]                        

def _process_chunk(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    enc_list: List[BatchEncoding],
    max_new_tokens: int,
    pad_id: int,
    device: torch.device,
) -> List[GenResult]:
    """
    Collate `enc_list`, run greedy generation, and return one GenResult
    per item: (generated_ids, probs, decoded_text)
    """
    in_seqs  = [e["input_ids"].squeeze(0)    for e in enc_list]          # 1‑D each
    attn_ms  = [e["attention_mask"].squeeze(0) for e in enc_list]

    input_ids = pad_sequence(in_seqs,  batch_first=True, padding_value=pad_id)
    attn_mask = pad_sequence(attn_ms,  batch_first=True, padding_value=0)

    input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=0.0,
        )

    seqs   = out.sequences                                   # (B , P+T)
    scores = torch.stack(out.scores, dim=1)                  # (B , T , |V|)

    P        = input_ids.size(1)                             # prompt length after padding
    gen_ids  = seqs[:, :]                                   # (B , P+T)
    probs    = torch.softmax(scores, dim=-1).to(torch.float16).cpu()   # (B , T , |V|)

    # Decode generated tokens to text (skip special tokens)
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids] 

    return [
        (gid.cpu(), pr, txt)
        for gid, pr, txt in zip(gen_ids, probs, texts)
    ]

def _flatten(tokenised: TokenisedPrompt) -> Tuple[List[BatchEncoding], List[Tuple[str | None, int]]]:
    """
    Flattens List[...] or Dict[str, List[...]] into a single list.
    Returns the flat list *and* a mapping so we can rebuild the
    original structure afterwards.
    """
    flat, index = [], []
    for k, lst in tokenised.items():
        for i, enc in enumerate(lst):
            flat.append(enc)
            index.append((k, i))
    return flat, index

def get_tokens_and_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_prompts: TokenisedPrompt,
    *,
    max_new_tokens: int = 128,
    micro_batch_size: int | None = None,
) -> Dict[str, List[GenResult]]: # GenResult = (tokens, probs, decoded_text)
    """
    Batched greedy generation returning (tokens, probs, decoded_text).
    """
    device   = next(model.parameters()).device
    pad_id   = model.config.pad_token_id or model.config.eos_token_id
    flat_enc, mapping = _flatten(tokenized_prompts)

    if not micro_batch_size or micro_batch_size <= 0:
        micro_batch_size = len(flat_enc)

    flat_results: List[GenResult] = []
    for start in range(0, len(flat_enc), micro_batch_size):
        chunk = flat_enc[start : start + micro_batch_size]
        print(f"Processing micro‑batch {start // micro_batch_size + 1}"
              f" of {len(flat_enc) // micro_batch_size} …", flush=True)

        flat_results.extend(
            _process_chunk(model, tokenizer, chunk, max_new_tokens, pad_id, device)
        )
        torch.cuda.empty_cache()

    # ── rebuild nested structure identical to tokenized_prompts ────────
    nested: Dict[str, List[GenResult]] = {}
    for (key, pos), res in zip(mapping, flat_results):
        nested.setdefault(key, []).append(res)
    return nested


# ----------------------------------------------------------------------
#--- Teacher forcing version of the above function
# ----------------------------------------------------------------------
def get_teacher_forcing_tokens_and_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    non_deactivated_token_and_logits: Dict[str, List[GenResult]],
    *,
    micro_batch_size: int = 32,
) -> Dict[str, List[GenResult]]:
    """
    Run teacher forcing with the deactivated model using tokens from non_deactivated_token_and_logits.
    Returns the same structure but with probabilities from the deactivated model.
    If sample_alternative_tokens=True, also samples what tokens the deactivated model would have chosen.
    """
    device = next(model.parameters()).device
    pad_id = model.config.pad_token_id or model.config.eos_token_id
    
    # Flatten the non_deactivated results to process in batches
    flat_items = []
    mapping = []
    
    for key, gen_results in non_deactivated_token_and_logits.items():
        for i, (tokens, probs, decoded_text) in enumerate(gen_results):
            flat_items.append((tokens, probs, decoded_text))
            mapping.append((key, i))
    
    # Process in micro-batches
    flat_results: List[GenResult] = []
    
    for start in range(0, len(flat_items), micro_batch_size):
        end = min(start + micro_batch_size, len(flat_items))
        chunk = flat_items[start:end]
        
        print(f"Processing teacher forcing micro-batch {start // micro_batch_size + 1}"
              f" of {(len(flat_items) + micro_batch_size - 1) // micro_batch_size} …", flush=True)
        
        # Process this chunk
        chunk_results = _process_teacher_forcing_chunk(
            model, tokenizer, chunk, pad_id, device
        )
        flat_results.extend(chunk_results)
        torch.cuda.empty_cache()
    
    # Rebuild nested structure identical to non_deactivated_token_and_logits
    nested: Dict[str, List[GenResult]] = {}
    for (key, pos), res in zip(mapping, flat_results):
        nested.setdefault(key, []).append(res)
    
    return nested


def _process_teacher_forcing_chunk(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    chunk: List[GenResult],  # List of (tokens, probs, decoded_text)
    pad_id: int,
    device: torch.device,
) -> List[GenResult]:
    """
    Process a chunk of teacher forcing items.
    """
    # Extract just the token sequences from the chunk
    token_sequences = [tokens for tokens, _, _ in chunk]
    
    # Pad sequences to same length
    padded_tokens = pad_sequence(token_sequences, batch_first=True, padding_value=pad_id)
    padded_tokens = padded_tokens.to(device)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_tokens != pad_id).long()
    
    with torch.no_grad():
        # Forward pass to get logits
        outputs = model(
            input_ids=padded_tokens,
            attention_mask=attention_mask,
            return_dict=True,
            temperature=0.0,  # Use temperature=0 for deterministic output
        )
        
        # Get probabilities from logits
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1).to(torch.float16).cpu()  # (batch_size, seq_len, vocab_size)
    
    # Process results for each item in the chunk
    results = []
    for i, (original_tokens, original_probs, original_decoded_text) in enumerate(chunk):
        # Extract probabilities for this sequence  excluding the prompt and the last token
        seq_len = original_probs.shape[0]           # == T
        prompt_len = original_tokens.shape[0] - seq_len   # == P

        start = prompt_len - 1 # logits that predict the generated tokens start at position (P‑1)
        end   = start + seq_len # avoid the last token logits

        item_probs = probs[i, start:end, :]       # shape (T, |V|)

        # Sample from the probability distribution at each position
        sampled_tokens = torch.multinomial(item_probs, num_samples=1).squeeze(-1)  # (seq_len,)
        
        # Decode the sampled tokens to see what the deactivated model would have generated
        sampled_decoded = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
            
        results.append((sampled_tokens, item_probs, sampled_decoded))
    
    return results