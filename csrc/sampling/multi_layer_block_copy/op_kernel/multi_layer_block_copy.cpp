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

#include "kernel_operator.h"

#include <cstdint>

using namespace AscendC;

namespace {

constexpr uint32_t COPY_BUFFER_NUM = 2;
constexpr uint32_t KEY_VALUE_COUNT = 2;

template <typename T>
__aicore__ inline __gm__ T* GetDynamicTensorAddress(uint32_t index,
                                                    GM_ADDR tensor_list) {
  __gm__ uint64_t* data_address =
      reinterpret_cast<__gm__ uint64_t*>(tensor_list);
  const uint64_t address_table_offset = *data_address;
  __gm__ uint64_t* address_table =
      data_address + (address_table_offset >> 3);
  return reinterpret_cast<__gm__ T*>(*(address_table + index));
}

class MultiLayerBlockCopyKernel {
 public:
  __aicore__ inline MultiLayerBlockCopyKernel() {}

  __aicore__ inline void Init(
      GM_ADDR key_caches, GM_ADDR value_caches, GM_ADDR block_mapping,
      const MultiLayerBlockCopyTilingData* tiling_data, TPipe* pipe) {
    key_cache_list_ = key_caches;
    value_cache_list_ = value_caches;
    block_mapping_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(block_mapping),
        static_cast<uint64_t>(tiling_data->num_pairs) * 2);

    num_pairs_ = tiling_data->num_pairs;
    core_num_ = tiling_data->core_num;
    chunks_per_block_ = tiling_data->chunks_per_block;
    block_bytes_ = tiling_data->block_bytes;
    tile_bytes_ = tiling_data->tile_bytes;
    total_tasks_ = tiling_data->total_tasks;

    pipe->InitBuffer(copy_queue_, COPY_BUFFER_NUM,
                     static_cast<uint32_t>(tile_bytes_));
  }

  __aicore__ inline void Process() {
    const uint64_t block_index = GetBlockIdx();
    for (uint64_t task_index = block_index; task_index < total_tasks_;
         task_index += core_num_) {
      CopyTask(task_index);
    }
  }

 private:
  __aicore__ inline void CopyTask(uint64_t task_index) {
    const uint32_t chunk_index =
        static_cast<uint32_t>(task_index % chunks_per_block_);
    task_index /= chunks_per_block_;
    const uint32_t pair_index =
        static_cast<uint32_t>(task_index % num_pairs_);
    task_index /= num_pairs_;
    const uint32_t key_value_index =
        static_cast<uint32_t>(task_index % KEY_VALUE_COUNT);
    const uint32_t layer_index =
        static_cast<uint32_t>(task_index / KEY_VALUE_COUNT);

    const int32_t source_block = block_mapping_.GetValue(pair_index * 2);
    const int32_t destination_block =
        block_mapping_.GetValue(pair_index * 2 + 1);
    event_t scalar_to_mte2 =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(scalar_to_mte2);
    WaitFlag<HardEvent::S_MTE2>(scalar_to_mte2);

    const GM_ADDR cache_list =
        key_value_index == 0 ? key_cache_list_ : value_cache_list_;
    GlobalTensor<uint8_t> cache;
    cache.SetGlobalBuffer(
        GetDynamicTensorAddress<uint8_t>(layer_index, cache_list));

    const uint64_t chunk_offset =
        static_cast<uint64_t>(chunk_index) * tile_bytes_;
    const uint64_t remaining_bytes = block_bytes_ - chunk_offset;
    const uint64_t copy_bytes =
        tile_bytes_ < remaining_bytes ? tile_bytes_ : remaining_bytes;
    const uint64_t source_offset =
        static_cast<uint64_t>(source_block) * block_bytes_ + chunk_offset;
    const uint64_t destination_offset =
        static_cast<uint64_t>(destination_block) * block_bytes_ + chunk_offset;

    LocalTensor<uint8_t> cache_local = copy_queue_.AllocTensor<uint8_t>();
    DataCopyExtParams copy_params{
        1, static_cast<uint32_t>(copy_bytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};
    DataCopyPad(cache_local, cache[source_offset], copy_params, pad_params);
    copy_queue_.EnQue(cache_local);

    cache_local = copy_queue_.DeQue<uint8_t>();
    DataCopyPad(cache[destination_offset], cache_local, copy_params);
    copy_queue_.FreeTensor(cache_local);
  }

  TQueBind<TPosition::VECIN, TPosition::VECOUT, COPY_BUFFER_NUM> copy_queue_;
  GlobalTensor<int32_t> block_mapping_;
  GM_ADDR key_cache_list_;
  GM_ADDR value_cache_list_;
  uint32_t num_pairs_;
  uint32_t core_num_;
  uint32_t chunks_per_block_;
  uint64_t block_bytes_;
  uint64_t tile_bytes_;
  uint64_t total_tasks_;
};

}  // namespace

extern "C" __global__ __aicore__ void multi_layer_block_copy(
    GM_ADDR key_caches, GM_ADDR value_caches, GM_ADDR block_mapping,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  GET_TILING_DATA(tiling_data, tiling);
  TPipe pipe;
  MultiLayerBlockCopyKernel kernel;
  kernel.Init(key_caches, value_caches, block_mapping, &tiling_data, &pipe);
  kernel.Process();
}
