import unittest

import torch

from kv_pages import PagePool


def _pool(num_blocks=4, block_size=4):
    return PagePool(
        num_blocks=num_blocks,
        block_size=block_size,
        n_layers=2,
        n_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )


class PagePoolTests(unittest.TestCase):
    def test_refuse_when_pages_are_gone(self):
        pool = _pool(num_blocks=2, block_size=4)
        self.assertIsNotNone(pool.try_alloc(2))
        self.assertIsNone(pool.try_alloc(1))
        self.assertEqual(pool.free_count(), 0)

    def test_release_then_alloc(self):
        pool = _pool(num_blocks=2, block_size=4)
        ids = pool.try_alloc(2)
        pool.release(ids, n_tokens=0)
        self.assertEqual(pool.free_count(), 2)
        self.assertIsNotNone(pool.try_alloc(2))

    def test_write_gather_roundtrip_across_pages(self):
        pool = _pool(num_blocks=4, block_size=4)
        ids = pool.try_alloc(pool.blocks_for(6))
        self.assertEqual(len(ids), 2)
        keys = [(torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8) + layer) for layer in range(2)]
        values = [tensor + 100 for tensor in keys]
        pool.write(ids, keys, values, 0)
        got = pool.gather(ids, 6)
        for layer in range(2):
            self.assertTrue(torch.equal(got[layer][0][0], keys[layer]))
            self.assertTrue(torch.equal(got[layer][1][0], values[layer]))
        self.assertEqual(pool.stats()["filled_tokens"], 6)

    def test_append_one_token_on_the_next_page(self):
        pool = _pool(num_blocks=3, block_size=4)
        ids = pool.try_alloc(1)
        keys = [torch.ones(2, 4, 8) * (layer + 1) for layer in range(2)]
        values = [torch.ones(2, 4, 8) * -1 for _ in range(2)]
        pool.write(ids, keys, values, 0)
        self.assertTrue(pool.ensure_slot(ids, num_tokens=4))
        self.assertEqual(len(ids), 2)
        new_k = [torch.full((2, 1, 8), 7.0 + layer) for layer in range(2)]
        new_v = [torch.full((2, 1, 8), 3.0) for _ in range(2)]
        pool.write(ids, new_k, new_v, logical_start=4)
        got = pool.gather(ids, 5)
        self.assertTrue(torch.equal(got[0][0][0, :, :4, :], keys[0]))
        self.assertTrue(torch.equal(got[0][0][0, :, 4:, :], new_k[0]))

    def test_ensure_slot_refuses_when_the_pool_is_empty(self):
        pool = _pool(num_blocks=1, block_size=4)
        ids = pool.try_alloc(1)
        pool.write(ids, [torch.zeros(2, 4, 8), torch.zeros(2, 4, 8)], [torch.zeros(2, 4, 8), torch.zeros(2, 4, 8)], 0)
        self.assertFalse(pool.ensure_slot(ids, num_tokens=4))
        self.assertEqual(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
