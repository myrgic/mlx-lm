# Copyright © 2024 Apple Inc.

"""Unit tests for BlockedKVCache.

These tests exercise the block-level KV cache added for the substrate's
block-hash provider interface (RFC-0006 / cog::B2).  They do not require a
loaded language model; all tests use small random tensors.
"""

import sys
import types
import importlib.util
import unittest

import mlx.core as mx


def _load_cache_module():
    """Load mlx_lm.models.cache without triggering the generate.py import
    (which requires a newer mlx version than may be installed locally)."""
    pkg = sys.modules.get("mlx_lm")
    if pkg is None:
        pkg = types.ModuleType("mlx_lm")
        pkg.__path__ = ["mlx_lm"]
        sys.modules["mlx_lm"] = pkg

    models_pkg = sys.modules.get("mlx_lm.models")
    if models_pkg is None:
        models_pkg = types.ModuleType("mlx_lm.models")
        models_pkg.__path__ = ["mlx_lm/models"]
        sys.modules["mlx_lm.models"] = models_pkg

    base_name = "mlx_lm.models.base"
    if base_name not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(
            base_name, "mlx_lm/models/base.py"
        )
        base_mod = importlib.util.module_from_spec(base_spec)
        sys.modules[base_name] = base_mod
        base_spec.loader.exec_module(base_mod)

    cache_name = "mlx_lm.models.cache"
    if cache_name not in sys.modules:
        cache_spec = importlib.util.spec_from_file_location(
            cache_name, "mlx_lm/models/cache.py"
        )
        cache_mod = importlib.util.module_from_spec(cache_spec)
        sys.modules[cache_name] = cache_mod
        cache_spec.loader.exec_module(cache_mod)

    return sys.modules["mlx_lm.models.cache"]


_cache = _load_cache_module()
BlockedKVCache = _cache.BlockedKVCache
BlockHash = _cache.BlockHash
_hash_block = _cache._hash_block


class TestBlockHash(unittest.TestCase):
    def test_repr(self):
        h = BlockHash(0xDEADBEEF)
        self.assertIn("0x", repr(h))

    def test_hash_block_determinism(self):
        h1 = _hash_block([1, 2, 3, 4], 0)
        h2 = _hash_block([1, 2, 3, 4], 0)
        self.assertEqual(h1, h2)

    def test_hash_block_parent_sensitivity(self):
        h_a = _hash_block([1, 2, 3, 4], 0)
        h_b = _hash_block([1, 2, 3, 4], 1)
        self.assertNotEqual(h_a, h_b)

    def test_hash_block_token_sensitivity(self):
        h_a = _hash_block([1, 2, 3, 4], 0)
        h_b = _hash_block([1, 2, 3, 5], 0)
        self.assertNotEqual(h_a, h_b)


class TestBlockedKVCacheBasic(unittest.TestCase):
    def _make_kv(self, S, B=1, H=2, D=16):
        return (
            mx.random.uniform(shape=(B, H, S, D)),
            mx.random.uniform(shape=(B, H, S, D)),
        )

    def test_empty_on_init(self):
        c = BlockedKVCache(block_size=4)
        self.assertTrue(c.empty())
        self.assertEqual(c.offset, 0)
        self.assertEqual(c.num_complete_blocks, 0)
        self.assertEqual(c.block_hashes, [])

    def test_partial_block_no_hashes(self):
        c = BlockedKVCache(block_size=4)
        k, v = self._make_kv(3)
        c.update_and_fetch(k, v, token_ids=[0, 1, 2])
        self.assertEqual(c.offset, 3)
        self.assertEqual(c.num_complete_blocks, 0)
        self.assertEqual(c.block_hashes, [])

    def test_single_complete_block(self):
        c = BlockedKVCache(block_size=4)
        k, v = self._make_kv(4)
        c.update_and_fetch(k, v, token_ids=[0, 1, 2, 3])
        self.assertEqual(c.num_complete_blocks, 1)
        self.assertEqual(len(c.block_hashes), 1)
        self.assertIsInstance(c.block_hashes[0], BlockHash)

    def test_multiple_complete_blocks(self):
        c = BlockedKVCache(block_size=4)
        k, v = self._make_kv(8)
        c.update_and_fetch(k, v, token_ids=list(range(8)))
        self.assertEqual(c.num_complete_blocks, 2)
        self.assertEqual(len(c.block_hashes), 2)

    def test_blocks_across_multiple_updates(self):
        c = BlockedKVCache(block_size=4)
        for i in range(8):
            k, v = self._make_kv(1)
            c.update_and_fetch(k, v, token_ids=[i])
        self.assertEqual(c.num_complete_blocks, 2)
        self.assertEqual(c.offset, 8)

    def test_returned_shape(self):
        c = BlockedKVCache(block_size=4)
        k, v = self._make_kv(7, B=2)
        rk, rv = c.update_and_fetch(k, v, token_ids=list(range(7)))
        self.assertEqual(rk.shape, (2, 2, 7, 16))
        self.assertEqual(rv.shape, (2, 2, 7, 16))

    def test_without_token_ids(self):
        """update_and_fetch must work when token_ids is omitted."""
        c = BlockedKVCache(block_size=4)
        k, v = self._make_kv(4)
        rk, rv = c.update_and_fetch(k, v)
        self.assertEqual(c.num_complete_blocks, 1)
        self.assertEqual(rk.shape, (1, 2, 4, 16))


class TestBlockedKVCacheHashes(unittest.TestCase):
    def _make_kv(self, S):
        return (
            mx.random.uniform(shape=(1, 2, S, 16)),
            mx.random.uniform(shape=(1, 2, S, 16)),
        )

    def test_same_tokens_same_hashes(self):
        """Two caches with the same token sequence must produce identical hashes."""
        tids = list(range(8))
        ca = BlockedKVCache(block_size=4)
        cb = BlockedKVCache(block_size=4)
        ka, va = self._make_kv(8)
        kb, vb = self._make_kv(8)
        ca.update_and_fetch(ka, va, token_ids=tids)
        cb.update_and_fetch(kb, vb, token_ids=tids)
        for ha, hb in zip(ca.block_hashes, cb.block_hashes):
            self.assertEqual(ha.value, hb.value)

    def test_different_tokens_different_hashes(self):
        tids_a = [0, 1, 2, 3]
        tids_b = [0, 1, 2, 99]
        ca = BlockedKVCache(block_size=4)
        cb = BlockedKVCache(block_size=4)
        ka, va = self._make_kv(4)
        kb, vb = self._make_kv(4)
        ca.update_and_fetch(ka, va, token_ids=tids_a)
        cb.update_and_fetch(kb, vb, token_ids=tids_b)
        self.assertNotEqual(ca.block_hashes[0].value, cb.block_hashes[0].value)

    def test_hash_chain(self):
        """Block hashes must form a Merkle chain (block[1].value depends on block[0])."""
        tids = list(range(8))
        c = BlockedKVCache(block_size=4)
        k, v = self._make_kv(8)
        c.update_and_fetch(k, v, token_ids=tids)
        h0 = c.block_hashes[0].value
        h1_expected = _hash_block(tids[4:8], h0)
        self.assertEqual(c.block_hashes[1].value, h1_expected)

    def test_shared_prefix_produces_shared_hashes(self):
        """Sequences sharing a prefix share the same block hashes for that prefix."""
        prefix = list(range(4))
        suffix_a = [10, 11, 12, 13]
        suffix_b = [20, 21, 22, 23]

        ca = BlockedKVCache(block_size=4)
        cb = BlockedKVCache(block_size=4)
        ka, va = self._make_kv(8)
        kb, vb = self._make_kv(8)
        ca.update_and_fetch(ka, va, token_ids=prefix + suffix_a)
        cb.update_and_fetch(kb, vb, token_ids=prefix + suffix_b)

        # Block 0 (prefix) must match
        self.assertEqual(ca.block_hashes[0].value, cb.block_hashes[0].value)
        # Block 1 (diverged) must differ
        self.assertNotEqual(ca.block_hashes[1].value, cb.block_hashes[1].value)

    def test_on_block_full_callback(self):
        events = []
        c = BlockedKVCache(block_size=4, on_block_full=lambda idx, h: events.append((idx, h)))
        k, v = mx.zeros((1, 2, 8, 8)), mx.zeros((1, 2, 8, 8))
        c.update_and_fetch(k, v, token_ids=list(range(8)))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[1][0], 1)
        self.assertIsInstance(events[0][1], BlockHash)


class TestBlockedKVCacheProtocol(unittest.TestCase):
    """Tests for _BaseCache protocol compliance."""

    def _filled_cache(self, n_tokens=8, block_size=4):
        c = BlockedKVCache(block_size=block_size)
        k = mx.random.uniform(shape=(1, 2, n_tokens, 16))
        v = mx.random.uniform(shape=(1, 2, n_tokens, 16))
        c.update_and_fetch(k, v, token_ids=list(range(n_tokens)))
        return c

    def test_is_trimmable(self):
        c = self._filled_cache()
        self.assertTrue(c.is_trimmable())

    def test_trim_reduces_offset(self):
        c = self._filled_cache(8)
        trimmed = c.trim(4)
        self.assertEqual(trimmed, 4)
        self.assertEqual(c.offset, 4)

    def test_trim_updates_block_hashes(self):
        c = self._filled_cache(8, block_size=4)
        self.assertEqual(c.num_complete_blocks, 2)
        c.trim(4)
        # Exactly one block remains
        self.assertEqual(c.num_complete_blocks, 1)

    def test_trim_clamp_at_zero(self):
        c = self._filled_cache(4)
        trimmed = c.trim(100)
        self.assertEqual(trimmed, 4)
        self.assertEqual(c.offset, 0)

    def test_size(self):
        c = self._filled_cache(7)
        self.assertEqual(c.size(), 7)

    def test_nbytes(self):
        c = BlockedKVCache(block_size=4)
        self.assertEqual(c.nbytes, 0)
        k, v = mx.zeros((1, 2, 4, 16)), mx.zeros((1, 2, 4, 16))
        c.update_and_fetch(k, v)
        self.assertGreater(c.nbytes, 0)

    def test_state_roundtrip(self):
        c = self._filled_cache(8)
        k_orig, v_orig = c.state
        c2 = BlockedKVCache(block_size=4)
        c2.state = (k_orig, v_orig)
        self.assertEqual(c2.offset, 8)
        k2, v2 = c2.state
        self.assertTrue(mx.array_equal(k_orig, k2))
        self.assertTrue(mx.array_equal(v_orig, v2))

    def test_meta_state_roundtrip(self):
        c = self._filled_cache(8, block_size=4)
        meta = c.meta_state
        c2 = BlockedKVCache(block_size=4)
        c2.meta_state = meta
        self.assertEqual(c2.block_size, 4)
        self.assertEqual(c2.offset, 8)
        # num_complete_blocks should be reconstructed
        self.assertEqual(c2.num_complete_blocks, 2)

    def test_make_mask_delegates_to_base(self):
        """make_mask should return None for a single token with no offset."""
        c = BlockedKVCache(block_size=4)
        mask = c.make_mask(1, return_array=False, window_size=None)
        self.assertIsNone(mask)


class TestBlockedKVCacheFork(unittest.TestCase):
    def test_fork_shares_hashes(self):
        c = BlockedKVCache(block_size=4)
        k, v = mx.random.uniform(shape=(1, 2, 4, 16)), mx.random.uniform(shape=(1, 2, 4, 16))
        c.update_and_fetch(k, v, token_ids=[1, 2, 3, 4])
        forked = c.fork()
        self.assertEqual(forked.num_complete_blocks, 1)
        self.assertEqual(forked.block_hashes[0].value, c.block_hashes[0].value)

    def test_fork_independence(self):
        """Appending to fork must not affect parent."""
        c = BlockedKVCache(block_size=4)
        k, v = mx.random.uniform(shape=(1, 2, 4, 16)), mx.random.uniform(shape=(1, 2, 4, 16))
        c.update_and_fetch(k, v, token_ids=[1, 2, 3, 4])
        forked = c.fork()

        # Append different tokens to the fork
        kf, vf = mx.random.uniform(shape=(1, 2, 4, 16)), mx.random.uniform(shape=(1, 2, 4, 16))
        forked.update_and_fetch(kf, vf, token_ids=[10, 11, 12, 13])
        self.assertEqual(c.num_complete_blocks, 1)  # parent unchanged
        self.assertEqual(forked.num_complete_blocks, 2)

    def test_fork_divergent_hash_after_append(self):
        """Forked caches with different tokens produce different second-block hashes."""
        c = BlockedKVCache(block_size=4)
        k, v = mx.random.uniform(shape=(1, 2, 4, 16)), mx.random.uniform(shape=(1, 2, 4, 16))
        c.update_and_fetch(k, v, token_ids=[1, 2, 3, 4])

        fork_a = c.fork()
        fork_b = c.fork()

        ka, va = mx.random.uniform(shape=(1, 2, 4, 16)), mx.random.uniform(shape=(1, 2, 4, 16))
        fork_a.update_and_fetch(ka, va, token_ids=[10, 11, 12, 13])
        fork_b.update_and_fetch(ka, va, token_ids=[20, 21, 22, 23])

        # First block shared
        self.assertEqual(fork_a.block_hashes[0].value, fork_b.block_hashes[0].value)
        # Second block diverged
        self.assertNotEqual(fork_a.block_hashes[1].value, fork_b.block_hashes[1].value)


if __name__ == "__main__":
    unittest.main()
