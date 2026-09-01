# BeamSearchGroup custom operator

This operator is adapted from `xllm-ops` commit
`27034e9394d9063a8a4cfb1f70bcda0a1cad38c3` and retains the original
copyright and Apache-2.0 license headers.

The vLLM Ascend integration exposes the eager operator as
`torch.ops._C_ascend.npu_beam_search_group`. It is intentionally not registered
with a Meta implementation or captured by ACLGraph yet; sampling remains a
post-forward PyTorch operator call. The registered adapter supports the ReqLoop
decode contract (`current_step > 0` and `top_k == beam_width`); prefill remains
on ReqLoop's existing path.
