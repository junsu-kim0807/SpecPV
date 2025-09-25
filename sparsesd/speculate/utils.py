import random
# typing
from typing import List

import torch

from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

TOPK = 10  # topk for sparse tree

def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def reset_tree_mode(
    model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def chunked_prefilling(
    input_ids, model, full_past_key_values, eagle_past_key_values, logits_processor, chunk_size=1024
):
    seqlen = input_ids.size(1)
    for start in range(0, seqlen, chunk_size):
        end = min(start + chunk_size, seqlen)
        chunk = input_ids[:, start:end]
        outputs, orig, hidden_states = model(
            chunk,
            full_past_key_values=full_past_key_values,
            output_orig=True,
        )

        # Clone the output hidden states
        if model.use_eagle3:
            ea_device = model.ea_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [
                    x.to(ea_device) for x in outputs["hidden_states"]
                ]
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        if end < seqlen:
            # chunked eagle prefilling
            eagle_input_ids = input_ids[:, start+1:end+1]
            eagle_hidden = model.ea_layer(
                hidden_states, 
                input_ids=eagle_input_ids, 
                past_key_values=eagle_past_key_values,
                use_cache=True
            )

    # generate draft tokens
    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]

    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
        model.ea_layer.tree_draft(
            hidden_states, input_ids, model.base_model.lm_head, eagle_past_key_values, logits_processor
        )
    )
    return (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        orig,
        hidden_states,
        token,
    )


def should_partial_verify(partial_past_key_values, total_tokens):
    enabled = partial_past_key_values.enabled
    initialized = partial_past_key_values.retrieval_initialized
    capacity = (partial_past_key_values.get_seq_length() + total_tokens + 1 <= partial_past_key_values.cache_config.total_budget) # with a sampled token
    return (enabled and initialized and capacity)


# @record_time("verify")
def tree_decoding(
    model,
    tree_candidates,
    full_past_key_values,
    partial_past_key_values,
    tree_position_ids,
    input_ids,
    retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    # set kv cache
    missing_lens = 0
    total_tokens = model.ea_layer.total_tokens
    if should_partial_verify(partial_past_key_values, total_tokens):
        # print('patial verify')
        full_cache = None
        full_past_key_values.enabled = False
        partial_cache = partial_past_key_values
    elif partial_past_key_values.enabled:
        full_cache = full_past_key_values
        full_cache.enabled = True
        partial_cache = partial_past_key_values
        partial_cache.reset_buffer()
        kv_len = full_cache.get_seq_length()
        if kv_len < input_ids.shape[1]:
            # cat un-full verifed tokens
            tree_candidates = torch.cat((input_ids[:, kv_len:], tree_candidates), dim=1)
            missing_pos = torch.arange(kv_len, input_ids.shape[1], device=input_ids.device, dtype=position_ids.dtype).unsqueeze(0)
            position_ids = torch.cat((missing_pos, position_ids), dim=1)
            missing_lens = input_ids.shape[1] - kv_len
        partial_past_key_values.global_verified_lens = input_ids.shape[1]
    else:
        full_cache = full_past_key_values
        full_cache.enabled = True
        partial_cache = None

    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        full_past_key_values=full_cache,
        partial_past_key_values=partial_cache,
        position_ids=position_ids,
    )
    tree_logits = tree_logits[:, missing_lens:, :]

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [
                x.to(ea_device) for x in outputs["hidden_states"]
            ]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        # 调整，无放回采样
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.inference_mode()
def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    full_past_key_values,
    partial_past_key_values,
    draft_past_key_values,
    model,
    hidden_state_new,
    sample_p,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [
            input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
        ],
        dim=-1,
    )
    
    # Roll back full_kv, partial_kv and draft_kv
    if full_past_key_values.enabled:
        # roll back full 
        for layer_cache in full_past_key_values.layers:
            key_layer = layer_cache.keys
            value_layer = layer_cache.values
            tgt_len = select_indices.numel()
            # 1. copy data
            dst_k = key_layer[:, :, prev_input_len:prev_input_len + tgt_len, :]
            dst_v = value_layer[:, :, prev_input_len:prev_input_len + tgt_len, :]
            src_k = key_layer[:, :, select_indices.to(key_layer.device), :].clone()
            src_v = value_layer[:, :, select_indices.to(value_layer.device), :].clone()
            dst_k.copy_(src_k, non_blocking=True)
            dst_v.copy_(src_v, non_blocking=True)
            # 2. set zero
            key_layer[:, :, prev_input_len + tgt_len:, :].zero_()
            value_layer[:, :, prev_input_len + tgt_len:, :].zero_()

    if partial_past_key_values.enabled:
        # Set retrieval_initialized here to ensure consistency:
        # both update_inference_inputs() and tree_decoding() will observe
        # the same should_partial_verify state after the first initialization.
        partial_past_key_values.retrieval_initialized = True
        # roll back partial
        for layer_idx, (key_cache, value_cache) in enumerate(zip(partial_past_key_values.key_cache, partial_past_key_values.value_cache)):
            buf_k = key_cache["buffer"]
            buf_v = value_cache["buffer"]
            tgt_len = select_indices.numel()
            # 0. get indices
            verified_len = partial_past_key_values.verified_lens[layer_idx]
            local_indices = select_indices - prev_input_len + verified_len
            local_indices = local_indices.to(buf_k.device)
            # 1. copy data
            src_k = buf_k[:, :, local_indices, :].clone()
            src_v = buf_v[:, :, local_indices, :].clone()
            buf_k[:, :, verified_len:verified_len+tgt_len, :].copy_(src_k, non_blocking=True)
            buf_v[:, :, verified_len:verified_len+tgt_len, :].copy_(src_v, non_blocking=True)
            # 2. set zero 
            buf_k[:, :, verified_len+tgt_len:, :].zero_()
            buf_v[:, :, verified_len+tgt_len:, :].zero_()
            # 3. update verified_lens
            partial_past_key_values.verified_lens[layer_idx] += tgt_len

    # roll back draft_kv
    rollback_len = model.ea_layer.stable_length
    for layer_cache in draft_past_key_values.layers:
        key_layer = layer_cache.keys
        value_layer = layer_cache.values
        key_layer[:, :, rollback_len:, :].zero_()
        value_layer[:, :, rollback_len:, :].zero_()

    # prepare hidden states and input_ids for draft
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[
        :, best_candidate, : accept_length + 1
    ]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]

    # draft    
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
        model.ea_layer.tree_draft(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
            head=model.base_model.lm_head,
            past_key_values=draft_past_key_values,
            logits_processor=logits_processor,
        )
    )

    new_token += accept_length + 1

    return (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
    )


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
