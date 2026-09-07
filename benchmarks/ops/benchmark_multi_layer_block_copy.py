"""Microbenchmark for pair-based all-layer KV block copy."""

import argparse
import statistics

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


BEAM_WIDTHS = [16, 32, 128, 256, 1024]


def _measure(operation, warmup, repetitions):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for iteration in range(warmup + repetitions):
        start.record()
        operation()
        end.record()
        torch.npu.synchronize()
        if iteration >= warmup:
            samples.append(start.elapsed_time(end) * 1000)
    samples.sort()
    p99_index = min(len(samples) - 1, int(len(samples) * 0.99))
    return statistics.median(samples), samples[p99_index]


def _make_inputs(beam_width, num_layers, layout, block_size, num_heads, head_dim):
    num_blocks = beam_width * 2
    if layout == "BSHD":
        cache_shape = (num_blocks, block_size, num_heads, head_dim)
    else:
        cache_shape = (num_blocks, num_heads, block_size, head_dim)
    key_caches = [torch.randn(cache_shape, dtype=torch.float16).npu() for _ in range(num_layers)]
    value_caches = [torch.randn(cache_shape, dtype=torch.float16).npu() for _ in range(num_layers)]
    source_blocks = torch.arange(beam_width, dtype=torch.int32).npu() // 4
    destination_blocks = torch.arange(beam_width, dtype=torch.int32).npu() + beam_width
    mapping = torch.stack((source_blocks, destination_blocks), dim=1).contiguous()
    return key_caches, value_caches, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--layout", choices=("BSHD", "BHSD"), default="BSHD")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()

    print("beam_width,baseline_p50_us,fused_p50_us,fused_p99_us,speedup")
    for beam_width in BEAM_WIDTHS:
        key_caches, value_caches, mapping = _make_inputs(
            beam_width,
            args.num_layers,
            args.layout,
            args.block_size,
            args.num_heads,
            args.head_dim,
        )

        source_indices = mapping[:, 0].long()
        destination_indices = mapping[:, 1].long()

        def baseline_operation():
            for key_cache, value_cache in zip(key_caches, value_caches):
                key_cache[destination_indices] = key_cache[source_indices]
                value_cache[destination_indices] = value_cache[source_indices]

        def fused_operation():
            torch.ops._C_ascend.npu_multi_layer_block_copy(
                key_caches,
                value_caches,
                mapping,
            )

        baseline_p50_us, _ = _measure(
            baseline_operation,
            args.warmup,
            args.repetitions,
        )
        fused_p50_us, fused_p99_us = _measure(
            fused_operation,
            args.warmup,
            args.repetitions,
        )
        speedup = baseline_p50_us / fused_p50_us
        print(
            f"{beam_width},{baseline_p50_us:.3f},{fused_p50_us:.3f},"
            f"{fused_p99_us:.3f},{speedup:.3f}"
        )


if __name__ == "__main__":
    main()
