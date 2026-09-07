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

#include "register/op_def_registry.h"

namespace ops {

class MultiLayerBlockCopy : public OpDef {
 public:
  explicit MultiLayerBlockCopy(const char* name) : OpDef(name) {
    this->Input("key_caches")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("value_caches")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("block_mapping")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat(
            {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("num_layers").Int();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(MultiLayerBlockCopy);

}  // namespace ops
