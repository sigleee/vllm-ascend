# BeamSearchGroup 单算子接入与验证

BeamSearchGroup 以 ACLNN 自定义算子接入 vLLM Ascend，通过
`torch.ops._C_ascend.npu_beam_search_group` 调用。当前仅支持 eager
单算子调用，不进入 ACLGraph；验证范围为 `current_step > 0` 且
`top_k == beam_width`。

## 1. 准备代码

```bash
git checkout dev_beam_search_group
git submodule update --init --recursive csrc/third_party/catlass
```

## 2. 仅启用 BeamSearchGroup

为缩短调试编译时间，临时将 `csrc/build_aclnn.sh` 中当前 SOC 分支的
`CUSTOM_OPS_ARRAY` 改为：

```bash
CUSTOM_OPS_ARRAY=(
    "beam_search_group"
)
```

该修改仅用于本地验证，不要提交。

## 3. 编译并安装

在 vLLM Ascend 仓库根目录执行：

```bash
rm -rf csrc/build csrc/build_out csrc/output
MAX_JOBS=4 python -m pip install -v --no-build-isolation -e .
```

构建完成后，自定义算子安装在：

```text
vllm_ascend/_cann_ops_custom/vendors/custom_transformer/
```

## 4. 验证注册

```bash
python - <<'PY'
import torch
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_beam_search_group")
print("BeamSearchGroup registration OK")
PY
```

## 5. 运行精度测试

```bash
pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_beam_search_group.py
```

验证结束后，恢复 `csrc/build_aclnn.sh` 中的完整算子列表。
