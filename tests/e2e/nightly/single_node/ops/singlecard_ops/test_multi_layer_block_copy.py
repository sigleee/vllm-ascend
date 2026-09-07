"""Eager accuracy tests for the MultiLayerBlockCopy custom operator."""

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


BEAM_WIDTHS = [16, 32, 128, 256, 1024]
NUM_LAYERS = 32
BLOCK_SIZE = 4
NUM_HEADS = 2
HEAD_DIM = 16
CHILDREN_PER_PARENT = 4


def _cache_shape(num_blocks, layout):
    if layout == "BSHD":
        return (num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
    return (num_blocks, NUM_HEADS, BLOCK_SIZE, HEAD_DIM)


@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
@pytest.mark.parametrize("layout", ["BSHD", "BHSD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multi_layer_block_copy_all_layers(beam_width, layout, dtype):
    """Copy repeated parent blocks through one launch for a 32-layer cache."""
    torch.manual_seed(2026 + beam_width)
    num_blocks = beam_width * 2
    shape = _cache_shape(num_blocks, layout)
    key_caches = [torch.randn(shape, dtype=dtype).npu() for _ in range(NUM_LAYERS)]
    value_caches = [torch.randn(shape, dtype=dtype).npu() for _ in range(NUM_LAYERS)]

    source_blocks = torch.arange(beam_width, dtype=torch.int32) // CHILDREN_PER_PARENT
    destination_blocks = torch.arange(beam_width, dtype=torch.int32) + beam_width
    block_mapping = torch.stack((source_blocks, destination_blocks), dim=1).npu()

    source_indices = source_blocks.long()
    expected_keys = [cache.cpu()[source_indices] for cache in key_caches]
    expected_values = [cache.cpu()[source_indices] for cache in value_caches]
    torch.ops._C_ascend.npu_multi_layer_block_copy(
        key_caches,
        value_caches,
        block_mapping,
    )
    torch.npu.synchronize()

    destination_indices = destination_blocks.long()
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(
            key_caches[layer].cpu()[destination_indices],
            expected_keys[layer],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            value_caches[layer].cpu()[destination_indices],
            expected_values[layer],
            rtol=0,
            atol=0,
        )


def test_multi_layer_block_copy_rejects_mismatched_layer_lists():
    shape = _cache_shape(32, "BSHD")
    key_caches = [torch.zeros(shape, dtype=torch.float16).npu()]
    value_caches = [torch.zeros(shape, dtype=torch.float16).npu() for _ in range(2)]
    block_mapping = torch.tensor([[0, 16]], dtype=torch.int32).npu()

    with pytest.raises(RuntimeError, match="same number of layers"):
        torch.ops._C_ascend.npu_multi_layer_block_copy(
            key_caches,
            value_caches,
            block_mapping,
        )
