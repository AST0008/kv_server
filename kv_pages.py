"""Fixed-size KV pages.

The model forward still sees one contiguous cache per request. These
pages are the memory the server owns: a request holds a block table,
and a new request is refused when the free list cannot cover its prompt.
"""

import threading

import torch


class PagePool:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device,
    ):
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.bytes_per_block = (
            2 * n_layers * n_kv_heads * block_size * head_dim * torch.empty((), dtype=dtype).element_size()
        )
        self.keys = torch.empty(
            n_layers, num_blocks, n_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        self.values = torch.empty_like(self.keys)
        self._free = list(range(num_blocks))
        self._free_set = set(self._free)
        self._lock = threading.Lock()
        self.filled_tokens = 0

    def blocks_for(self, n_tokens: int) -> int:
        if n_tokens <= 0:
            return 0
        return (n_tokens + self.block_size - 1) // self.block_size

    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    def try_alloc(self, n: int):
        """Return n physical block ids, or None if the free list is short."""
        if n < 0:
            raise ValueError("n must be >= 0")
        with self._lock:
            if n == 0:
                return []
            if len(self._free) < n:
                return None
            ids = self._free[-n:]
            del self._free[-n:]
            for block_id in ids:
                self._free_set.remove(block_id)
            return ids

    def release(self, block_ids, n_tokens: int = 0) -> None:
        if not block_ids and n_tokens == 0:
            return
        with self._lock:
            for block_id in block_ids:
                if block_id in self._free_set:
                    continue
                self._free_set.add(block_id)
                self._free.append(block_id)
            self.filled_tokens = max(0, self.filled_tokens - n_tokens)

    def ensure_slot(self, block_ids: list, num_tokens: int) -> bool:
        """Make sure the next token has a page. False means the pool is empty."""
        need = self.blocks_for(num_tokens + 1)
        extra_n = need - len(block_ids)
        if extra_n <= 0:
            return True
        extra = self.try_alloc(extra_n)
        if extra is None:
            return False
        block_ids.extend(extra)
        return True

    def write(self, block_ids, keys, values, logical_start: int) -> None:
        """Copy token KV into pages.

        keys[layer] and values[layer] are [heads, n_tokens, dim], already
        the real tokens (no batch dimension, no padding).
        """
        n_tokens = keys[0].shape[1]
        if n_tokens == 0:
            return
        pos = 0
        while pos < n_tokens:
            logical = logical_start + pos
            offset = logical % self.block_size
            take = min(self.block_size - offset, n_tokens - pos)
            phys = block_ids[logical // self.block_size]
            for layer in range(self.n_layers):
                self.keys[layer, phys, :, offset:offset + take, :] = keys[layer][:, pos:pos + take, :]
                self.values[layer, phys, :, offset:offset + take, :] = values[layer][:, pos:pos + take, :]
            pos += take
        with self._lock:
            self.filled_tokens += n_tokens

    def gather(self, block_ids, n_tokens: int):
        """Contiguous cache for one request: ((key, value), ...) with shape [1, heads, tokens, dim]."""
        if n_tokens <= 0:
            raise ValueError("n_tokens must be > 0")
        layers = []
        remaining = n_tokens
        # Walk blocks once so every layer reads the same spans.
        spans = []
        for phys in block_ids:
            if remaining <= 0:
                break
            take = min(self.block_size, remaining)
            spans.append((phys, take))
            remaining -= take
        if remaining > 0:
            raise RuntimeError("block table is shorter than num_tokens")
        for layer in range(self.n_layers):
            chunks_k = [self.keys[layer, phys, :, :take, :] for phys, take in spans]
            chunks_v = [self.values[layer, phys, :, :take, :] for phys, take in spans]
            key = torch.cat(chunks_k, dim=1).unsqueeze(0).contiguous()
            value = torch.cat(chunks_v, dim=1).unsqueeze(0).contiguous()
            layers.append((key, value))
        return tuple(layers)

    def stats(self) -> dict:
        with self._lock:
            free = len(self._free)
            filled = self.filled_tokens
        used = self.num_blocks - free
        kv_bytes = used * self.bytes_per_block
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "free_blocks": free,
            "used_blocks": used,
            "filled_tokens": filled,
            "bytes_per_block": self.bytes_per_block,
            "bytes_per_token": self.bytes_per_block // self.block_size,
            "kv_bytes": kv_bytes,
            "kv_mb": round(kv_bytes / (1024 ** 2), 2),
        }
