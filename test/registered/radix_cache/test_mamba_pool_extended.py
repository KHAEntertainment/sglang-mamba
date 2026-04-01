import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=30, suite="stage-b-test-small-1-gpu-amd")


def _make_pool(
    max_num_reqs=10, mamba_cache_size=20, max_context_len=128, enable_extra_buffer=False
):
    device = get_device()
    num_layers = 48
    global_interval = 4
    full_attention_layer_ids = [
        i for i in range(global_interval - 1, num_layers, global_interval)
    ]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=4096,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
    return HybridReqToTokenPool(
        size=max_num_reqs,
        mamba_size=mamba_cache_size,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=enable_extra_buffer,
        speculative_num_draft_tokens=3,
    )


def _make_req():
    return Req(
        rid=0,
        origin_input_text="",
        origin_input_ids=[],
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


class TestMambaPoolExtended(unittest.TestCase):
    def setUp(self):
        self.max_num_reqs = 4
        self.mamba_cache_size = 6
        self.pool = _make_pool(self.max_num_reqs, self.mamba_cache_size)
        self.initial_req_size = self.pool.available_size()
        self.initial_mamba_size = self.pool.mamba_pool.available_size()

    def tearDown(self):
        self.assertEqual(
            self.pool.mamba_pool.available_size(),
            self.initial_mamba_size,
            "Mamba pool available_size did not return to initial — possible leak",
        )

    def test_pool_exhaustion(self):
        # Allocate up to max_num_reqs (the limiting factor for req pool)
        reqs = [_make_req() for _ in range(self.max_num_reqs)]
        for req in reqs:
            req.rid = id(req)
        for req in reqs:
            self.pool.alloc([req])
        self.assertEqual(self.pool.available_size(), 0)
        extra_req = _make_req()
        self.assertIsNone(self.pool.alloc([extra_req]))
        for req in reqs:
            self.pool.free_mamba_cache(req)
            self.pool.free(req)

    def test_mamba_pool_reuse_on_no_free(self):
        req = _make_req()
        self.pool.alloc([req])
        self.assertEqual(
            self.pool.mamba_pool.available_size(), self.initial_mamba_size - 1
        )
        self.pool.free(req)
        self.assertEqual(self.pool.available_size(), self.initial_req_size)
        self.assertEqual(
            self.pool.mamba_pool.available_size(), self.initial_mamba_size - 1
        )
        self.pool.alloc([req])
        self.assertEqual(
            self.pool.mamba_pool.available_size(), self.initial_mamba_size - 1
        )
        self.pool.free_mamba_cache(req)
        self.pool.free(req)

    def test_mamba_state_dtype_override(self):
        """SGLANG_MAMBA_SSM_DTYPE override produces bfloat16 temporal states in the pool tensors."""
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            pool = _make_pool()
        # Verify temporal state tensor at a known mamba layer is bfloat16
        mamba_layers = [
            i
            for i in range(48)
            if i not in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        ]
        layer_cache = pool.mamba_pool.mamba2_layer_cache(mamba_layers[0])
        self.assertEqual(layer_cache.temporal.dtype, torch.bfloat16)

    def test_get_mamba_indices_mapping(self):
        req = _make_req()
        self.pool.alloc([req])
        self.assertIsNotNone(req.mamba_pool_idx)
        idx_tensor = self.pool.get_mamba_indices(req.req_pool_idx)
        self.assertIsNotNone(idx_tensor)
        self.pool.free_mamba_cache(req)
        self.pool.free(req)

    def test_enable_mamba_extra_buffer_false(self):
        pool_no_extra = _make_pool(enable_extra_buffer=False)
        req = _make_req()
        pool_no_extra.alloc([req])
        self.assertIsNone(getattr(req, "mamba_ping_pong_track_buffer", None))
        pool_no_extra.free_mamba_cache(req)
        pool_no_extra.free(req)


if __name__ == "__main__":
    unittest.main()
