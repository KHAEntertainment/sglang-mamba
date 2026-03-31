from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=20, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=20, suite="stage-b-test-small-1-gpu-amd")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)


def _make_forward_metadata(num_seqs=4, device="cpu"):
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    mamba_cache_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
    return ForwardMetadata(
        query_start_loc=query_start_loc,
        mamba_cache_indices=mamba_cache_indices,
    )


class TestMamba2Metadata(unittest.TestCase):

    def test_prepare_decode_pure_decode_batch(self):
        N = 4
        seq_lens = torch.ones(N, dtype=torch.int32)
        fwd_meta = _make_forward_metadata(num_seqs=N)
        result = Mamba2Metadata.prepare_decode(fwd_meta, seq_lens, is_target_verify=False, draft_token_num=1)
        self.assertEqual(result.num_prefills, 0)
        self.assertEqual(result.num_decodes, N)
        self.assertEqual(result.num_prefill_tokens, 0)
        self.assertIsNone(result.mixed_metadata)

    def test_prepare_mixed_prefill_only(self):
        N = 3
        query_start_loc = torch.tensor([0, 5, 10, 15], dtype=torch.int32)
        mamba_cache_indices = torch.arange(N, dtype=torch.int32)
        fwd_meta = ForwardMetadata(query_start_loc=query_start_loc, mamba_cache_indices=mamba_cache_indices)
        forward_batch = MagicMock()
        forward_batch.extend_num_tokens = 15
        forward_batch.extend_seq_lens = [5] * N
        forward_batch.extend_seq_lens_cpu = [5] * N
        forward_batch.extend_prefix_lens = torch.zeros(N, dtype=torch.int32)
        forward_batch.seq_lens = torch.tensor([5] * N, dtype=torch.int32)
        forward_batch.spec_info = None
        forward_batch.forward_mode = MagicMock()
        forward_batch.forward_mode.is_target_verify.return_value = False
        chunk_size = 8
        result = Mamba2Metadata.prepare_mixed(fwd_meta, chunk_size, forward_batch)
        self.assertEqual(result.num_prefills, N)
        self.assertEqual(result.num_decodes, 0)
        self.assertEqual(result.num_prefill_tokens, 15)
        self.assertIsNotNone(result.mixed_metadata)
        self.assertFalse(result.mixed_metadata.prep_initial_states)

    def test_chunk_indices_offsets_correctness(self):
        query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32)
        chunk_size = 8
        total_seqlens = 10
        chunk_indices, chunk_offsets = Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
            query_start_loc, chunk_size, total_seqlens)
        expected_indices = torch.tensor([0, 0, 1], dtype=torch.int32)
        expected_offsets = torch.tensor([0, 5, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(chunk_indices, expected_indices),
                        f"chunk_indices mismatch: got {chunk_indices}, expected {expected_indices}")
        self.assertTrue(torch.equal(chunk_offsets, expected_offsets),
                        f"chunk_offsets mismatch: got {chunk_offsets}, expected {expected_offsets}")

    def test_has_initial_states_flag(self):
        N = 4
        # query_start_loc must match extend_seq_lens=[5]*4 → cumsum [0,5,10,15,20]
        query_start_loc = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32)
        mamba_cache_indices = torch.arange(N, dtype=torch.int32)
        fwd_meta = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
        )
        forward_batch = MagicMock()
        forward_batch.extend_num_tokens = 20
        forward_batch.extend_seq_lens = [5] * N
        forward_batch.extend_seq_lens_cpu = [5] * N
        forward_batch.extend_prefix_lens = torch.tensor([10, 5, 0, 0], dtype=torch.int32)
        forward_batch.seq_lens = torch.tensor([5] * N, dtype=torch.int32)
        forward_batch.spec_info = None
        forward_batch.forward_mode = MagicMock()
        forward_batch.forward_mode.is_target_verify.return_value = False
        chunk_size = 8
        result = Mamba2Metadata.prepare_mixed(fwd_meta, chunk_size, forward_batch)
        self.assertIsNotNone(result.mixed_metadata)
        expected_has_initial = torch.tensor([True, True, False, False])
        self.assertTrue(torch.equal(result.mixed_metadata.has_initial_states, expected_has_initial),
                        f"has_initial_states: got {result.mixed_metadata.has_initial_states}")
        self.assertTrue(result.mixed_metadata.prep_initial_states)

    def test_mamba_cache_indices_preserved(self):
        N = 3
        indices = torch.tensor([7, 3, 11], dtype=torch.int32)
        fwd_meta = ForwardMetadata(
            query_start_loc=torch.arange(N + 1, dtype=torch.int32),
            mamba_cache_indices=indices,
        )
        seq_lens = torch.ones(N, dtype=torch.int32)
        result = Mamba2Metadata.prepare_decode(fwd_meta, seq_lens, is_target_verify=False, draft_token_num=1)
        self.assertTrue(torch.equal(result.mamba_cache_indices, indices),
                        f"mamba_cache_indices changed: got {result.mamba_cache_indices}")


if __name__ == "__main__":
    unittest.main()
