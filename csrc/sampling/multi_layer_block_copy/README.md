# MultiLayerBlockCopy 多层 KV cache 搬运方案

## 1. 目标与结论

生成式推荐的 SID beam search 在父 beam 分叉后，需要把父序列的尾部
KV block 复制到各子 beam 新分配的 block。原路径在 Python 中逐层执行
K/V 高级索引，模型有多少层就会产生多少轮算子下发；beam width 小于
AIV 核数时，单层任务也不能提供足够并行度。

本方案新增 ACLNN 自定义算子 `MultiLayerBlockCopy`，由
`torch.ops._C_ascend.npu_multi_layer_block_copy` 调用。一次调用直接接收
全部层的 K cache 与 V cache `Tensor[]`，把任务展开到
`layer × K/V × src-dst pair × block chunk`，在 Ascend 910B/910C 上通过
一次 kernel 下发完成全模型 cache 搬运。

## 2. 接口

```python
torch.ops._C_ascend.npu_multi_layer_block_copy(
    key_caches: list[Tensor],
    value_caches: list[Tensor],
    block_mapping: Tensor,
) -> None
```

- `key_caches` / `value_caches`：长度均为 `num_layers`。每个 tensor 为连续
  ND tensor，shape 支持：
  - `(num_blocks, block_size, head_num, head_dim)`（BSHD）；
  - `(num_blocks, head_num, block_size, head_dim)`（BHSD）。
- 全部层的 shape 和 dtype 必须相同；K/V shape 也必须相同。
- cache dtype 支持 FP16、BF16、INT8。
- `block_mapping`：NPU 上连续的 INT32 tensor，shape 为
  `(num_pairs, 2)`；每行依次为 `(src_block_id, dst_block_id)`。
- 算子原地更新 cache，不产生输出 tensor。

两个 layout 的一个 block 都是从第一维 block id 开始的一段连续内存，
因此 kernel 只需要使用 `prod(shape[1:]) * element_size` 作为 block 字节数，
无需为 BSHD/BHSD 分别编译模板。

当前版本沿用 pair 语义。调用方必须保证 block id 在
`[0, num_blocks)` 内、目的 block 互不重复，并且同次调用的目的 block
不作为其他 pair 的源 block。beam 分叉场景中，目的 block 是新分配的，
满足该约束。

## 3. Host 与 kernel 设计

### Host 侧

`AscendAttentionBackend.copy_blocks` 把各层 `(2, num_blocks, ...)` cache
拆成 K/V 两个 4D view 列表，并把 mapping 一次性转换为同设备的 INT32
连续 tensor。此前的逐层高级索引循环被一次自定义算子调用替代。

Torch binding 对设备、连续性、层数、shape、dtype 和 mapping shape 做
快速校验；不读取 NPU tensor 的 block id，避免在热路径引入 D2H 同步。

### Tiling 侧

单个 cache block 较大时按不超过 64 KiB 的 chunk 切分，实际 tile 还受
`UB / 2` 限制。任务总数为：

```text
total_tasks = num_layers * 2 * num_pairs * chunks_per_block
block_dim   = min(aiv_core_num, total_tasks)
```

每个 AIV 核以 grid-stride 方式处理任务。即使 `beam_width=16`，以 32 层
为例，在 block 不切片时也有 `32 × 2 × 16 = 1024` 个任务，足以占满
910B/910C 的 AIV 核。

### Device 侧

动态 `Tensor[]` 输入在 kernel 中表现为地址表。每个任务解析 layer、K/V、
pair 与 chunk，读出 `(src, dst)`，然后执行：

```text
GM[src block + chunk] -> UB -> GM[dst block + chunk]
```

数据按字节搬运，所以 FP16、BF16、INT8 共用同一份 kernel 逻辑；尾 chunk
通过 `DataCopyPad` 处理非 32-byte 对齐长度。算子不申请 workspace。

## 4. 正确性与性能验证

精度用例位于：

```text
tests/e2e/nightly/single_node/ops/singlecard_ops/test_multi_layer_block_copy.py
```

用例使用 32 层 K/V cache，同时覆盖：

- beam width：16、32、128、256、1024；
- cache layout：BSHD、BHSD；
- cache dtype：FP16、BF16；
- 一个父 block 对应 4 个子 block的重复源读取场景；
- K 与 V、所有层的逐元素一致性；
- K/V layer list 数量不一致的失败路径。

建议在 910B 与 910C 上分别记录以下基准指标：

- 原逐层路径与新算子的端到端 P50/P99 latency；
- kernel execution time 与 effective HBM bandwidth；
- AIV active core 数与利用率；
- mapping H2D 时间（mapping 原本就在 NPU 时单独记录零拷贝路径）。

仓库同时提供 `benchmarks/ops/benchmark_multi_layer_block_copy.py`，默认按
32 层输出五档 beam width 的原逐层路径 P50、新算子 P50/P99 与 speedup。

构建与测试：

```bash
# 调试时可把当前 SOC 的 CUSTOM_OPS_ARRAY 临时缩减为：
# CUSTOM_OPS_ARRAY=("multi_layer_block_copy")
MAX_JOBS=4 python -m pip install -v --no-build-isolation -e .

pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_multi_layer_block_copy.py

python benchmarks/ops/benchmark_multi_layer_block_copy.py
```

## 5. 后续：按 src block 分组

当前 pair 版本在一个父 beam 分叉为多个子 beam 时会重复从 GM 读取源
block。下一阶段可把 mapping 改为 CSR 风格：

```text
src_block_ids: [num_unique_src]
dst_block_ids: [num_pairs]
dst_cumsum:    [num_unique_src]
```

每个 `layer × K/V × src × chunk` 任务只把源 chunk 从 GM 读入 UB 一次，
再依次写到 `dst_cumsum` 指定区间内的多个目的 block。超大 fan-out 可以
按目的数量再次分片，在“源重复读取次数”和“并行任务数”之间做 tiling
权衡。该接口也能直接消费 `beam_search_group` 已输出的父 beam 分组与
prefix sum，避免 host 侧重新排序。
