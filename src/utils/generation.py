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

PromptLike = Union[str, List[str], Dict[str, List[str]]]
TokenisedPrompt = Union[BatchEncoding, List[BatchEncoding], Dict[str, List[BatchEncoding]]]

def template_tokenize_single_prompt(prompt: str, tokenizer: PreTrainedTokenizer, *, prompt_template: str = "no", tokenize_kwargs: Dict = None) -> BatchEncoding:
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    templated = apply_prompt_template(prompt, tokenizer, prompt_template)
    return tokenizer(templated, **tokenize_kwargs)

def template_tokenize_prompts(prompts: PromptLike, tokenizer: PreTrainedTokenizer, *, prompt_template: str = "no", tokenize_kwargs: Dict = None) -> TokenisedPrompt:
    if isinstance(prompts, str):
        return template_tokenize_single_prompt(prompts, tokenizer, prompt_template=prompt_template, tokenize_kwargs=tokenize_kwargs)
    elif isinstance(prompts, list):
        return [template_tokenize_single_prompt(p, tokenizer, prompt_template=prompt_template, tokenize_kwargs=tokenize_kwargs) for p in prompts]
    elif isinstance(prompts, dict):
        return {k: [template_tokenize_single_prompt(p, tokenizer, prompt_template=prompt_template, tokenize_kwargs=tokenize_kwargs) for p in v] for k, v in prompts.items()}
    else:
        raise TypeError(f"`prompts` must be str, List[str], or Dict[str, List[str]] (got {type(prompts)})")

GenResult      = Tuple[torch.LongTensor, torch.FloatTensor]       # (T,) , (T,)
GenerationLike = Union[GenResult, List[GenResult], Dict[str, List[GenResult]]]

def _flatten(tokenised: TokenisedPrompt) -> Tuple[List[BatchEncoding], List[Tuple[str | None, int]]]:
    """
    Flattens List[...] or Dict[str, List[...]] into a single list.
    Returns the flat list *and* a mapping so we can rebuild the
    original structure afterwards.
    """
    flat, index = [], []
    if isinstance(tokenised, dict):
        for k, lst in tokenised.items():
            for i, enc in enumerate(lst):
                flat.append(enc)
                index.append((k, i))
    elif isinstance(tokenised, list):
        for i, enc in enumerate(tokenised):
            flat.append(enc)
            index.append((None, i))
    else:                                   # single BatchEncoding
        flat.append(tokenised)
        index.append((None, 0))
    return flat, index


# ── helper: one micro‑batch forward pass ─────────────────────────────────
def _process_chunk(
    model: PreTrainedModel,
    enc_list: List[BatchEncoding],
    max_new_tokens: int,
    pad_id: int,
    device: torch.device,
) -> List[GenResult]:
    """
    Collate `enc_list`, run greedy generation, and return one
    GenResult per item (in the same order).
    """
    in_seqs = [e["input_ids"].squeeze(0)   for e in enc_list]   # 1‑D each
    attn_ms = [e["attention_mask"].squeeze(0) for e in enc_list]

    input_ids = pad_sequence(in_seqs, batch_first=True, padding_value=pad_id)
    attn_mask = pad_sequence(attn_ms, batch_first=True, padding_value=0)

    input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seqs   = out.sequences                                # (B , P+T)
    scores = torch.stack(out.scores, dim=1)               # (B , T , |V|)

    P = input_ids.size(1)                                 # padded prompt len
    gen_ids = seqs[:, P:]                                 # (B , T)
    logp    = torch.log_softmax(scores, dim=-1)           # (B , T , |V|)
    gen_lp  = logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B , T)

    return [(gid.cpu(), glp.cpu()) for gid, glp in zip(gen_ids, gen_lp)]

# ── public API ───────────────────────────────────────────────────────────
def get_tokens_and_logits(
    model: PreTrainedModel,
    tokenized_prompts: TokenisedPrompt,
    *,
    max_new_tokens: int = 128,
    micro_batch_size: int | None = None,
) -> GenerationLike:
    """
    Batched greedy generation with per‑token log‑probs.

    Returns `GenerationLike` shaped identically to `tokenized_prompts`,
    but each leaf item is a `(gen_ids, gen_logprobs)` tuple.
    """
    device   = next(model.parameters()).device
    pad_id   = model.config.pad_token_id or model.config.eos_token_id
    flat_enc, mapping = _flatten(tokenized_prompts)

    if micro_batch_size is None or micro_batch_size <= 0:
        micro_batch_size = len(flat_enc)                  # single big batch

    flat_results: List[GenResult] = []
    for start in range(0, len(flat_enc), micro_batch_size):
        chunk = flat_enc[start : start + micro_batch_size]
        print(f"Processing micro-batch {start // micro_batch_size + 1} of {len(flat_enc) // micro_batch_size + 1}...", flush=True)
        flat_results.extend(
            _process_chunk(model, chunk, max_new_tokens, pad_id, device)
        )
        torch.cuda.empty_cache()  # ⇦ uncomment if you need aggressive GC

    # ── rebuild original nested structure ───────────────────────────────
    if isinstance(tokenized_prompts, dict):
        nested: Dict[str, List[GenResult]] = {}
        for (key, pos), res in zip(mapping, flat_results):
            nested.setdefault(key, []).append(res)
        return nested
    elif isinstance(tokenized_prompts, list):
        ordered = [None] * len(mapping)                   # type: ignore
        for (_, idx), res in zip(mapping, flat_results):
            ordered[idx] = res
        return ordered                                    # type: ignore
    else:
        return flat_results[0]

