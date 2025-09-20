import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import StaticCache, OffloadedStaticCache
from dataclasses import dataclass

class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


@dataclass
class CacheConfig:
    block_size: int = 16              
    n_sink_blocks: int = 4             
    n_retrieval_blocks: int = 256      
    n_buffer_blocks: int = 4         
    n_spec_tokens_buf: int = 100       
    max_batch_size: int = 1            

    @property
    def sink_size(self) -> int:
        return self.n_sink_blocks * self.block_size

    @property
    def retrieval_size(self) -> int:
        return self.n_retrieval_blocks * self.block_size
    
    @property
    def buffer_size(self) -> int:
        return self.n_buffer_blocks * self.block_size

    @property
    def total_budget(self) -> int:
        return self.sink_size + self.retrieval_size + self.buffer_size + self.n_spec_tokens_buf


class PartialKVCache(Cache):
    def __init__(self, cache_config, model_config, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        # config
        self.cache_config = cache_config
        self.model_config = model_config
        self.dtype = dtype
        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
        self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)

        self.key_cache: list[dict[str, torch.Tensor]] = []
        self.value_cache: list[dict[str, torch.Tensor]] = []
        self.verified_lens = []
        self.current_lens = []

        # build per-layer cache
        for _ in range(self.num_layers):
            # allocate one big tensor [batch, heads, total_budget, head_dim]
            total_budget = (
                cache_config.sink_size
                + cache_config.retrieval_size
                + cache_config.buffer_size
            )
            cache_shape = (cache_config.max_batch_size, self.num_kv_heads, total_budget, self.head_dim)

            key_storage = torch.zeros(cache_shape, dtype=dtype, device=device)
            value_storage = torch.zeros(cache_shape, dtype=dtype, device=device)

            # offsets
            offset = 0
            sink_slice = slice(offset, offset + cache_config.sink_size); offset += cache_config.sink_size
            retri_slice = slice(offset, offset + cache_config.retrieval_size); offset += cache_config.retrieval_size
            buffer_slice = slice(offset, offset + cache_config.buffer_size)

            # record partitions
            self.key_cache.append({
                "sink": key_storage[:, :, sink_slice, :],
                "retrieval": key_storage[:, :, retri_slice, :],
                "buffer": key_storage[:, :, buffer_slice, :],
                # no spec anymore
            })
            self.value_cache.append({
                "sink": value_storage[:, :, sink_slice, :],
                "retrieval": value_storage[:, :, retri_slice, :],
                "buffer": value_storage[:, :, buffer_slice, :],
            })
            self.verified_lens.append(0)
            self.current_lens.append(0)

    def summary_key_states(self):
        pass

    def add_to_sink(self, kv_blocks):
        pass

    def refresh_retrieval(self, query, full_cache_index):
        pass

    def update(self, new_tokens_kv):
        """
        Append new generated tokens' KV into the spec buffer.
        Return the active KV subset (sink + retrieval + buffer + spec_buffer)
        to be used for inference in the next step.
        """
        pass

    def spec_update(self, candidate_tokens_kv):
        """
        After partial verification, resolve candidate KV entries:
        - Accepted tokens are committed into the buffer partition.
        - Rejected tokens are discarded.
        """
        pass

    def get_seq_length():
        pass

    def clear(self):
        pass


# from pytorch_memlab import profile
# @profile
def initialize_past_key_values(model, draft_model, max_length=8192, offloading=True):
    config = model.config
    
    # init full kv cache
    full_past_key_values = StaticCache(
        config=config,
        max_cache_len=max_length,
        offloading=offloading
    )
    
    # init partial kv cache
    partial_cache_config = CacheConfig()
    partial_past_key_values = PartialKVCache(cache_config=partial_cache_config, model_config=config, dtype=model.dtype)

    # init draft kv cache
    draft_past_key_values = StaticCache(
        config=draft_model.config,
        max_cache_len=max_length,
    )

    return full_past_key_values, partial_past_key_values, draft_past_key_values
