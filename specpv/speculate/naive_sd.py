from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from .profile import record_time
from . import utils as sd_utils  # for optional logits_processor hooks


def _build_stop_ids(*, tokenizer, is_llama3: bool) -> list[int]:
    stop_ids: list[int] = []
    if is_llama3:
        # Llama-3 style stop token
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        stop_ids.append(int(eot_id))

    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))

    # Remove duplicates while keeping deterministic ordering.
    return sorted(set(stop_ids))


@record_time("naive_sd_propose")
@torch.no_grad()
def propose_tokens_greedy(
    *,
    draft_model,
    input_ids: torch.Tensor,
    spec_k: int,
    max_length: int,
    logits_processor=None,
) -> tuple[list[int], list[float]]:
    """
    Default propose function:
    - Greedily roll `spec_k` tokens with the draft model.
    - Returns:
      - proposed: list of token ids [t1..tK]
      - q_token_probs: list of q(ti | prefix) probabilities (float)
    """
    device = input_ids.device
    temp_ids = input_ids

    proposed: list[int] = []
    q_token_probs: list[float] = []

    for _ in range(spec_k):
        if temp_ids.shape[1] >= max_length:
            break
        out = draft_model(input_ids=temp_ids)
        next_logits = out.logits[:, -1, :]  # [1,vocab]
        if logits_processor is not None:
            processed = logits_processor(None, next_logits)
            if isinstance(processed, (tuple, list)):
                processed = processed[0]
            next_logits = processed
        q_probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.argmax(q_probs, dim=-1, keepdim=True)  # [1,1]
        tok = int(next_token.item())
        proposed.append(tok)
        q_token_probs.append(float(q_probs[0, tok].item()))
        temp_ids = torch.cat([temp_ids, next_token], dim=1)

    return proposed, q_token_probs


@record_time("naive_sd_verify")
@torch.no_grad()
def vanilla_speculative_decode(
    *,
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_length: int,
    is_llama3: bool,
    spec_k: int = 8,
    propose_fn: Optional[
        Callable[..., tuple[list[int], list[float]]]
    ] = None,
    target_logits_processor=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Vanilla speculative decoding (accept/reject) using generic HF causal LMs.

    This is draft-free for AR (method=ar), and draft-required for naive.
    Official "speculate" code paths are separate; this file is purely a
    lightweight accept/reject speculative decoder.

    Hook for "draft tree":
    - `propose_fn` can be replaced with a custom proposer that uses the
      project's draft tree (e.g., DraftAdapter.tree_draft) and returns:
        (proposed_tokens, q_token_probs)
    """
    start_time = time.time()
    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    output_ids = input_ids.clone()

    stop_ids = _build_stop_ids(tokenizer=tokenizer, is_llama3=is_llama3)

    if propose_fn is None:
        propose_fn = propose_tokens_greedy

    generated = 0
    accept_lengths: list[int] = []
    accepted_counts_per_pos = [0 for _ in range(spec_k)]
    proposed_counts_per_pos = [0 for _ in range(spec_k)]

    # KV-cache based fast path is only guaranteed when we use greedy proposing.
    use_kv_cache = propose_fn == propose_tokens_greedy

    if use_kv_cache:
        try:
            draft_out = draft_model(input_ids=output_ids, use_cache=True)
            target_out = target_model(input_ids=output_ids, use_cache=True)
            draft_past = getattr(draft_out, "past_key_values", None)
            target_past = getattr(target_out, "past_key_values", None)
            if draft_past is None or target_past is None:
                raise RuntimeError("No past_key_values returned")

            logits_draft = draft_out.logits[:, -1, :]  # [1,vocab]
            logits_target = target_out.logits[:, -1, :]  # [1,vocab]

            while generated < max_new_tokens and output_ids.shape[1] < max_length:
                accept_len = 0
                stop_reached = False
                rejected = False

                for pos in range(spec_k):
                    if generated >= max_new_tokens or output_ids.shape[1] >= max_length:
                        break

                    # 1) Draft proposes greedily from current draft logits.
                    q_probs = torch.softmax(logits_draft, dim=-1)
                    tok = int(torch.argmax(q_probs, dim=-1).item())
                    q_tok_prob = float(q_probs[0, tok].item())
                    proposed_counts_per_pos[pos] += 1

                    # 2) Target verifies using current target logits.
                    if target_logits_processor is not None:
                        processed = target_logits_processor(None, logits_target)
                        if isinstance(processed, (tuple, list)):
                            processed = processed[0]
                        p_logits = processed
                    else:
                        p_logits = logits_target
                    p_probs = torch.softmax(p_logits, dim=-1)
                    p_tok_prob = float(p_probs[0, tok].item())

                    # Acceptance probability a = min(1, p/q)
                    if q_tok_prob <= 0:
                        a = 1.0
                    else:
                        a = min(1.0, p_tok_prob / q_tok_prob)

                    if float(torch.rand(1).item()) <= a:
                        # Accept proposed token and advance both KV caches.
                        output_ids = torch.cat(
                            [
                                output_ids,
                                torch.tensor([[tok]], device=device, dtype=torch.long),
                            ],
                            dim=1,
                        )
                        generated += 1
                        accept_len += 1
                        accepted_counts_per_pos[pos] += 1

                        if tok in stop_ids:
                            stop_reached = True
                            break

                        next_tok_tensor = torch.tensor(
                            [[tok]], device=device, dtype=torch.long
                        )
                        draft_out = draft_model(
                            input_ids=next_tok_tensor,
                            past_key_values=draft_past,
                            use_cache=True,
                        )
                        draft_past = draft_out.past_key_values
                        logits_draft = draft_out.logits[:, -1, :]

                        target_out = target_model(
                            input_ids=next_tok_tensor,
                            past_key_values=target_past,
                            use_cache=True,
                        )
                        target_past = target_out.past_key_values
                        logits_target = target_out.logits[:, -1, :]
                    else:
                        # Reject: append deterministic argmax from target distribution.
                        rejected = True
                        next_tok_tensor = torch.argmax(p_probs, dim=-1, keepdim=True)
                        next_tok = int(next_tok_tensor.item())
                        output_ids = torch.cat(
                            [
                                output_ids,
                                torch.tensor(
                                    [[next_tok]], device=device, dtype=torch.long
                                ),
                            ],
                            dim=1,
                        )
                        generated += 1

                        if next_tok in stop_ids:
                            stop_reached = True

                        next_tok_tensor = torch.tensor(
                            [[next_tok]], device=device, dtype=torch.long
                        )
                        draft_out = draft_model(
                            input_ids=next_tok_tensor,
                            past_key_values=draft_past,
                            use_cache=True,
                        )
                        draft_past = draft_out.past_key_values
                        logits_draft = draft_out.logits[:, -1, :]

                        target_out = target_model(
                            input_ids=next_tok_tensor,
                            past_key_values=target_past,
                            use_cache=True,
                        )
                        target_past = target_out.past_key_values
                        logits_target = target_out.logits[:, -1, :]
                        break

                # If we accepted all K proposed tokens, we must also generate one
                # additional "bonus" token from the target model distribution.
                if (
                    (not rejected)
                    and (accept_len == spec_k)
                    and (not stop_reached)
                    and (generated < max_new_tokens)
                    and (output_ids.shape[1] < max_length)
                ):
                    bonus_logits = logits_target
                    if target_logits_processor is not None:
                        processed = target_logits_processor(None, bonus_logits)
                        if isinstance(processed, (tuple, list)):
                            processed = processed[0]
                        bonus_logits = processed
                    bonus_probs = torch.softmax(bonus_logits, dim=-1)
                    bonus_next_tok = int(torch.argmax(bonus_probs, dim=-1).item())

                    output_ids = torch.cat(
                        [
                            output_ids,
                            torch.tensor(
                                [[bonus_next_tok]], device=device, dtype=torch.long
                            ),
                        ],
                        dim=1,
                    )
                    generated += 1

                    if bonus_next_tok in stop_ids:
                        stop_reached = True
                    else:
                        # Advance both KV caches by one token.
                        bonus_next_tok_tensor = torch.tensor(
                            [[bonus_next_tok]], device=device, dtype=torch.long
                        )
                        draft_out = draft_model(
                            input_ids=bonus_next_tok_tensor,
                            past_key_values=draft_past,
                            use_cache=True,
                        )
                        draft_past = draft_out.past_key_values
                        logits_draft = draft_out.logits[:, -1, :]

                        target_out = target_model(
                            input_ids=bonus_next_tok_tensor,
                            past_key_values=target_past,
                            use_cache=True,
                        )
                        target_past = target_out.past_key_values
                        logits_target = target_out.logits[:, -1, :]

                accept_lengths.append(accept_len)
                if stop_reached:
                    break

            total_time = time.time() - start_time
            new_token = int(output_ids.shape[-1] - prompt_len)
            avg_accept_length = (
                sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
            )

            acceptance_rate_per_pos: list[float] = []
            for pos in range(spec_k):
                denom = proposed_counts_per_pos[pos]
                if denom > 0:
                    acceptance_rate_per_pos.append(accepted_counts_per_pos[pos] / denom)
                else:
                    acceptance_rate_per_pos.append(0.0)

            throughput = (new_token / total_time) if total_time > 0 else 0.0
            metrics = {
                "avg_accept_length": float(avg_accept_length),
                "acceptance_rate_per_pos": acceptance_rate_per_pos,
                "new_token": float(new_token),
                "total_time": float(total_time),
                "throughput": float(throughput),
            }
            return output_ids, metrics
        except Exception:
            # Fallback to legacy slow implementation.
            use_kv_cache = False

    # Legacy slow path (no KV cache / full recomputation).
    while generated < max_new_tokens and output_ids.shape[1] < max_length:
        proposed, q_token_probs = propose_fn(
            draft_model=draft_model,
            input_ids=output_ids,
            spec_k=spec_k,
            max_length=max_length,
        )
        if not proposed:
            break
        proposed_len = len(proposed)
        rejected = False
        for pos in range(min(spec_k, proposed_len)):
            proposed_counts_per_pos[pos] += 1

        proposed_ids = torch.tensor(proposed, device=device, dtype=torch.long).unsqueeze(0)  # [1,K]
        base_ids = output_ids
        verify_ids = torch.cat([base_ids, proposed_ids], dim=1)

        target_out = target_model(input_ids=verify_ids)
        target_logits = target_out.logits  # [1, L+K, vocab]

        stop_reached = False
        accept_len = 0

        for i, tok in enumerate(proposed):
            if generated >= max_new_tokens or output_ids.shape[1] >= max_length:
                break

            # target at index (base_len-1 + i) predicts verify proposed token i
            pos = base_ids.shape[1] + i - 1
            p_logits = target_logits[:, pos, :]
            if target_logits_processor is not None:
                processed = target_logits_processor(None, p_logits)
                if isinstance(processed, (tuple, list)):
                    processed = processed[0]
                p_logits = processed
            p_probs = torch.softmax(p_logits, dim=-1)
            p_tok_prob = float(p_probs[0, tok].item())
            q_tok_prob = q_token_probs[i]

            # Acceptance probability a = min(1, p/q)
            if q_tok_prob <= 0:
                a = 1.0
            else:
                a = min(1.0, p_tok_prob / q_tok_prob)

            if float(torch.rand(1).item()) <= a:
                output_ids = torch.cat(
                    [output_ids, torch.tensor([[tok]], device=device, dtype=torch.long)], dim=1
                )
                generated += 1
                accept_len += 1
                if i < spec_k:
                    accepted_counts_per_pos[i] += 1
                if tok in stop_ids:
                    stop_reached = True
                    break
            else:
                # Reject: append deterministic argmax from target distribution.
                rejected = True
                next_token = torch.argmax(p_probs, dim=-1, keepdim=True)  # [1,1]
                next_tok = int(next_token.item())
                output_ids = torch.cat(
                    [output_ids, torch.tensor([[next_tok]], device=device, dtype=torch.long)], dim=1
                )
                generated += 1
                if next_tok in stop_ids:
                    stop_reached = True
                break

        # If we accepted all proposed tokens, generate one additional bonus token
        # from the target model distribution.
        if (
            (not rejected)
            and (accept_len == proposed_len)
            and (not stop_reached)
            and (generated < max_new_tokens)
            and (output_ids.shape[1] < max_length)
        ):
            base_len = base_ids.shape[1]
            # After consuming proposed_len tokens, the next token logits are at:
            # index base_len + proposed_len - 1 (causal LM next-token position).
            pos_bonus = base_len + proposed_len - 1
            bonus_logits = target_logits[:, pos_bonus, :]
            if target_logits_processor is not None:
                processed = target_logits_processor(None, bonus_logits)
                if isinstance(processed, (tuple, list)):
                    processed = processed[0]
                bonus_logits = processed
            bonus_probs = torch.softmax(bonus_logits, dim=-1)
            bonus_next_tok = int(torch.argmax(bonus_probs, dim=-1).item())

            output_ids = torch.cat(
                [
                    output_ids,
                    torch.tensor([[bonus_next_tok]], device=device, dtype=torch.long),
                ],
                dim=1,
            )
            generated += 1

            if bonus_next_tok in stop_ids:
                stop_reached = True

        accept_lengths.append(accept_len)
        if stop_reached:
            break

    total_time = time.time() - start_time
    new_token = int(output_ids.shape[-1] - prompt_len)
    avg_accept_length = (
        sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
    )

    acceptance_rate_per_pos: list[float] = []
    for pos in range(spec_k):
        denom = proposed_counts_per_pos[pos]
        if denom > 0:
            acceptance_rate_per_pos.append(accepted_counts_per_pos[pos] / denom)
        else:
            acceptance_rate_per_pos.append(0.0)

    throughput = (new_token / total_time) if total_time > 0 else 0.0
    metrics = {
        "avg_accept_length": float(avg_accept_length),
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
        "new_token": float(new_token),
        "total_time": float(total_time),
        "throughput": float(throughput),
    }
    return output_ids, metrics

