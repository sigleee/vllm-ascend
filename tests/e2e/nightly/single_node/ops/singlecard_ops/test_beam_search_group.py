"""Eager accuracy tests for the BeamSearchGroup custom operator."""

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _reference(log_probs, top_tokens, top_probs, sequence, current_step, top_k):
    beam_width = top_tokens.shape[1]
    request_num = log_probs.shape[0] // beam_width
    candidate_scores = (
        log_probs.reshape(request_num, beam_width, 1)
        + top_probs.reshape(request_num, beam_width, beam_width)
    ).reshape(request_num, beam_width * beam_width)
    selected_scores, flat_indices = torch.topk(candidate_scores, k=top_k, dim=1, sorted=True)
    selected_parents = torch.div(flat_indices, beam_width, rounding_mode="floor")
    selected_offsets = torch.remainder(flat_indices, beam_width)
    top_tokens_3d = top_tokens.reshape(request_num, beam_width, beam_width)
    selected_tokens = torch.gather(
        torch.gather(
            top_tokens_3d,
            dim=1,
            index=selected_parents.unsqueeze(-1).expand(-1, -1, beam_width),
        ),
        dim=2,
        index=selected_offsets.unsqueeze(-1),
    ).squeeze(-1)

    out_tokens = torch.empty((request_num, top_k), dtype=torch.int32)
    out_parents = torch.empty_like(out_tokens)
    out_scores = torch.empty((request_num, top_k), dtype=torch.float32)
    out_prefix = torch.empty((request_num, beam_width), dtype=torch.int32)
    out_sequence = torch.empty((request_num, top_k, current_step + 1), dtype=torch.int32)
    for request_index in range(request_num):
        parents = selected_parents[request_index].tolist()
        order = torch.tensor(sorted(range(top_k), key=lambda index: (parents[index], index)))
        grouped_parents = selected_parents[request_index].index_select(0, order)
        grouped_tokens = selected_tokens[request_index].index_select(0, order)
        grouped_scores = selected_scores[request_index].index_select(0, order)
        counts = torch.bincount(grouped_parents, minlength=beam_width)
        out_tokens[request_index] = grouped_tokens
        out_parents[request_index] = grouped_parents.to(torch.int32) + request_index * beam_width
        out_scores[request_index] = grouped_scores
        out_prefix[request_index] = (
            torch.cumsum(counts, dim=0).to(torch.int32) + request_index * beam_width
        )
        parent_sequence = sequence[request_index].index_select(0, grouped_parents)
        out_sequence[request_index, :, :current_step] = parent_sequence[:, :current_step]
        out_sequence[request_index, :, current_step] = grouped_tokens
    return out_tokens, out_parents, out_scores, out_prefix, out_sequence


@pytest.mark.parametrize("request_num", [1, 2, 8, 48])
def test_beam_search_group_eager(request_num):
    torch.manual_seed(2026 + request_num)
    beam_width = 8
    current_step = 3
    top_k = beam_width
    num_sequences = request_num * beam_width
    log_probs = torch.randn((num_sequences, 1), dtype=torch.float32)
    top_probs = torch.randn((num_sequences, beam_width), dtype=torch.float32)
    top_tokens = torch.arange(num_sequences * beam_width, dtype=torch.int32).reshape(
        num_sequences, beam_width
    )
    sequence = torch.randint(
        0, 32000, (request_num, beam_width, current_step + 1), dtype=torch.int32
    )
    expected = _reference(log_probs, top_tokens, top_probs, sequence, current_step, top_k)

    actual = torch.ops._C_ascend.npu_beam_search_group(
        log_probs.npu(),
        top_tokens.npu(),
        top_probs.npu(),
        sequence.npu(),
        current_step,
        top_k,
    )
    torch.npu.synchronize()
    actual = tuple(tensor.cpu() for tensor in actual)

    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor.dtype.is_floating_point:
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-4, rtol=0)
        else:
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
