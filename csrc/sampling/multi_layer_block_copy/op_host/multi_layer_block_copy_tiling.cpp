/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "multi_layer_block_copy_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint64_t DMA_ALIGNMENT_BYTES = 32;
constexpr uint64_t MAX_TILE_BYTES = 64 * 1024;
constexpr uint32_t COPY_BUFFER_NUM = 2;
constexpr uint32_t KEY_VALUE_COUNT = 2;

uint32_t GetElementSize(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
      return 2;
    case ge::DT_INT8:
      return 1;
    default:
      return 0;
  }
}

bool ShapesEqual(const gert::Shape& lhs, const gert::Shape& rhs) {
  if (lhs.GetDimNum() != rhs.GetDimNum()) {
    return false;
  }
  for (size_t dim = 0; dim < lhs.GetDimNum(); ++dim) {
    if (lhs.GetDim(dim) != rhs.GetDim(dim)) {
      return false;
    }
  }
  return true;
}

ge::graphStatus MultiLayerBlockCopyTiling(gert::TilingContext* context) {
  auto* platform_info = context->GetPlatformInfo();
  OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
  auto platform = platform_ascendc::PlatformAscendC(platform_info);

  auto* attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t* num_layers_attr = attrs->GetAttrPointer<int64_t>(0);
  OPS_CHECK(num_layers_attr == nullptr || *num_layers_attr <= 0 ||
                static_cast<uint64_t>(*num_layers_attr) >
                    std::numeric_limits<uint32_t>::max(),
            OPS_LOG_E(context->GetNodeName(),
                      "num_layers must be greater than zero."),
            return ge::GRAPH_FAILED);
  const uint32_t num_layers = static_cast<uint32_t>(*num_layers_attr);

  auto* first_key_shape = context->GetDynamicInputShape(0, 0);
  auto* first_value_shape = context->GetDynamicInputShape(1, 0);
  auto* first_key_desc = context->GetDynamicInputDesc(0, 0);
  auto* first_value_desc = context->GetDynamicInputDesc(1, 0);
  auto* mapping_shape_ptr = context->GetDynamicInputShape(2, 0);
  OPS_CHECK(first_key_shape == nullptr || first_value_shape == nullptr ||
                first_key_desc == nullptr || first_value_desc == nullptr ||
                mapping_shape_ptr == nullptr,
            OPS_LOG_E(context->GetNodeName(), "Input metadata is missing."),
            return ge::GRAPH_FAILED);

  const auto key_shape = first_key_shape->GetOriginShape();
  const auto value_shape = first_value_shape->GetOriginShape();
  const auto mapping_shape = mapping_shape_ptr->GetOriginShape();
  OPS_CHECK(key_shape.GetDimNum() != 4 || value_shape.GetDimNum() != 4,
            OPS_LOG_E(context->GetNodeName(),
                      "Each key/value cache must be a 4D tensor."),
            return ge::GRAPH_FAILED);
  OPS_CHECK(!ShapesEqual(key_shape, value_shape),
            OPS_LOG_E(context->GetNodeName(),
                      "Key and value cache shapes must match."),
            return ge::GRAPH_FAILED);
  OPS_CHECK(key_shape.GetDim(0) <= 0,
            OPS_LOG_E(context->GetNodeName(),
                      "The cache must contain at least one block."),
            return ge::GRAPH_FAILED);
  OPS_CHECK(mapping_shape.GetDimNum() != 2 || mapping_shape.GetDim(1) != 2,
            OPS_LOG_E(context->GetNodeName(),
                      "block_mapping must have shape [num_pairs, 2]."),
            return ge::GRAPH_FAILED);
  OPS_CHECK(first_key_desc->GetDataType() != first_value_desc->GetDataType(),
            OPS_LOG_E(context->GetNodeName(),
                      "Key and value cache dtypes must match."),
            return ge::GRAPH_FAILED);

  const uint32_t element_size = GetElementSize(first_key_desc->GetDataType());
  OPS_CHECK(element_size == 0,
            OPS_LOG_E(context->GetNodeName(), "Unsupported cache dtype."),
            return ge::GRAPH_FAILED);

  uint64_t block_elements = 1;
  for (size_t dim = 1; dim < key_shape.GetDimNum(); ++dim) {
    const int64_t key_dim = key_shape.GetDim(dim);
    const int64_t value_dim = value_shape.GetDim(dim);
    OPS_CHECK(key_dim <= 0 || key_dim != value_dim,
              OPS_LOG_E(context->GetNodeName(),
                        "Key and value cache block shapes must match."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(block_elements >
                  std::numeric_limits<uint64_t>::max() /
                      static_cast<uint64_t>(key_dim),
              OPS_LOG_E(context->GetNodeName(), "Cache block size overflow."),
              return ge::GRAPH_FAILED);
    block_elements *= static_cast<uint64_t>(key_dim);
  }
  OPS_CHECK(block_elements >
                std::numeric_limits<uint64_t>::max() / element_size,
            OPS_LOG_E(context->GetNodeName(), "Cache block byte size overflow."),
            return ge::GRAPH_FAILED);
  const uint64_t block_bytes = block_elements * element_size;

  for (uint32_t layer = 0; layer < num_layers; ++layer) {
    auto* key_layer_shape = context->GetDynamicInputShape(0, layer);
    auto* value_layer_shape = context->GetDynamicInputShape(1, layer);
    auto* key_layer_desc = context->GetDynamicInputDesc(0, layer);
    auto* value_layer_desc = context->GetDynamicInputDesc(1, layer);
    OPS_CHECK(key_layer_shape == nullptr || value_layer_shape == nullptr ||
                  key_layer_desc == nullptr || value_layer_desc == nullptr,
              OPS_LOG_E(context->GetNodeName(),
                        "num_layers does not match the dynamic input count."),
              return ge::GRAPH_FAILED);
    const auto current_key_shape = key_layer_shape->GetOriginShape();
    const auto current_value_shape = value_layer_shape->GetOriginShape();
    OPS_CHECK(!ShapesEqual(current_key_shape, key_shape) ||
                  !ShapesEqual(current_value_shape, value_shape) ||
                  key_layer_desc->GetDataType() !=
                      first_key_desc->GetDataType() ||
                  value_layer_desc->GetDataType() !=
                      first_value_desc->GetDataType(),
              OPS_LOG_E(context->GetNodeName(),
                        "All cache layers must have identical shapes and dtypes."),
              return ge::GRAPH_FAILED);
  }

  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  uint64_t tile_bytes = std::min(MAX_TILE_BYTES, ub_size / COPY_BUFFER_NUM);
  tile_bytes = tile_bytes / DMA_ALIGNMENT_BYTES * DMA_ALIGNMENT_BYTES;
  OPS_CHECK(tile_bytes == 0,
            OPS_LOG_E(context->GetNodeName(),
                      "Not enough UB space for block copy."),
            return ge::GRAPH_FAILED);

  const uint64_t chunks_per_block =
      (block_bytes + tile_bytes - 1) / tile_bytes;
  const uint64_t num_pairs =
      static_cast<uint64_t>(mapping_shape.GetDim(0));
  OPS_CHECK(num_pairs == 0 ||
                num_pairs > std::numeric_limits<uint32_t>::max() ||
                chunks_per_block > std::numeric_limits<uint32_t>::max(),
            OPS_LOG_E(context->GetNodeName(),
                      "Pair or chunk count is outside the supported range."),
            return ge::GRAPH_FAILED);
  OPS_CHECK(chunks_per_block >
                std::numeric_limits<uint64_t>::max() / num_pairs /
                    KEY_VALUE_COUNT / num_layers,
            OPS_LOG_E(context->GetNodeName(), "Task count overflow."),
            return ge::GRAPH_FAILED);
  const uint64_t total_tasks = chunks_per_block * num_pairs *
                               KEY_VALUE_COUNT * num_layers;
  const uint32_t aiv_num = platform.GetCoreNumAiv();
  OPS_CHECK(aiv_num == 0,
            OPS_LOG_E(context->GetNodeName(), "No AIV core is available."),
            return ge::GRAPH_FAILED);
  const uint32_t core_num = static_cast<uint32_t>(
      std::min<uint64_t>(static_cast<uint64_t>(aiv_num), total_tasks));

  MultiLayerBlockCopyTilingData tiling_data;
  tiling_data.set_num_layers(num_layers);
  tiling_data.set_num_pairs(static_cast<uint32_t>(num_pairs));
  tiling_data.set_core_num(core_num);
  tiling_data.set_chunks_per_block(
      static_cast<uint32_t>(chunks_per_block));
  tiling_data.set_block_bytes(block_bytes);
  tiling_data.set_tile_bytes(tile_bytes);
  tiling_data.set_total_tasks(total_tasks);

  tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                           context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());
  size_t* workspace_sizes = context->GetWorkspaceSizes(1);
  workspace_sizes[0] = 0;
  context->SetBlockDim(core_num);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(MultiLayerBlockCopy).Tiling(MultiLayerBlockCopyTiling);

}  // namespace optiling
