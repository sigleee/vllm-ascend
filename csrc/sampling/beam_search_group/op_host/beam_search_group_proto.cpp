/* Copyright 2025 xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "beam_search_group_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* log_probs_shape = context->GetInputShape(0);
  const gert::Shape* top_tokens_shape = context->GetInputShape(1);
  const gert::Shape* sequence_shape = context->GetInputShape(3);
  auto* attrs = context->GetAttrs();
  if (log_probs_shape == nullptr || top_tokens_shape == nullptr ||
      sequence_shape == nullptr || attrs == nullptr) {
    return GRAPH_FAILED;
  }
  const int* top_k_attr = attrs->GetAttrPointer<int>(1);
  if (top_k_attr == nullptr) {
    return GRAPH_FAILED;
  }

  const int64_t num_sequences = log_probs_shape->GetDim(0);
  const int64_t beam_width = top_tokens_shape->GetDim(1);
  const int64_t request_num =
      (num_sequences < 0 || beam_width <= 0) ? -1 : num_sequences / beam_width;
  const int64_t top_k = static_cast<int64_t>(*top_k_attr);

  for (size_t output_index = 0; output_index < 3; ++output_index) {
    gert::Shape* output_shape = context->GetOutputShape(output_index);
    if (output_shape == nullptr) {
      return GRAPH_FAILED;
    }
    output_shape->SetDimNum(2);
    output_shape->SetDim(0, request_num);
    output_shape->SetDim(1, top_k);
  }

  gert::Shape* prefix_sums_shape = context->GetOutputShape(3);
  gert::Shape* output_sequence_shape = context->GetOutputShape(4);
  if (prefix_sums_shape == nullptr || output_sequence_shape == nullptr) {
    return GRAPH_FAILED;
  }
  prefix_sums_shape->SetDimNum(2);
  prefix_sums_shape->SetDim(0, request_num);
  prefix_sums_shape->SetDim(1, beam_width);
  output_sequence_shape->SetDimNum(3);
  output_sequence_shape->SetDim(0, request_num);
  output_sequence_shape->SetDim(1, top_k);
  output_sequence_shape->SetDim(2, sequence_shape->GetDim(2));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BeamSearchGroup).InferShape(InferShape);
}  // namespace ge
