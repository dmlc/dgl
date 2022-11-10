/**
 *  Copyright (c) 2021-2022 by Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * @file nccl_api.h
 * @brief Wrapper around NCCL routines.
 */

#ifndef DGL_RUNTIME_CUDA_NCCL_API_H_
#define DGL_RUNTIME_CUDA_NCCL_API_H_

#ifdef DGL_USE_NCCL
#include "nccl.h"
#else
// if not compiling with NCCL, this class will only support communicators of
// size 1.
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;
typedef int ncclComm_t;
#endif

#include <dgl/runtime/object.h>

#include <string>

namespace dgl {
namespace runtime {
namespace cuda {

class NCCLUniqueId : public runtime::Object {
 public:
  NCCLUniqueId();

  static constexpr const char* _type_key = "cuda.NCCLUniqueId";
  DGL_DECLARE_OBJECT_TYPE_INFO(NCCLUniqueId, Object);

  ncclUniqueId Get() const;

  std::string ToString() const;

  void FromString(const std::string& str);

 private:
  ncclUniqueId id_;
};

DGL_DEFINE_OBJECT_REF(NCCLUniqueIdRef, NCCLUniqueId);

class NCCLCommunicator : public runtime::Object {
 public:
  NCCLCommunicator(int size, int rank, ncclUniqueId id);

  ~NCCLCommunicator();

  // disable copying
  NCCLCommunicator(const NCCLCommunicator& other) = delete;
  NCCLCommunicator& operator=(const NCCLCommunicator& other);

  ncclComm_t Get();

  /**
   * @brief Perform an all-to-all communication.
   *
   * @param send The continous array of data to send.
   * @param recv The continous array of data to recieve.
   * @param count The size of data to send to each rank.
   * @param stream The stream to operate on.
   */
  template <typename IdType>
  void AllToAll(
      const IdType* send, IdType* recv, int64_t count, cudaStream_t stream);

  /**
   * @brief Perform an all-to-all variable sized communication.
   *
   * @tparam DType The type of value to send.
   * @param send The arrays of data to send.
   * @param send_prefix The prefix of each array to send.
   * @param recv The arrays of data to recieve.
   * @param recv_prefix The prefix of each array to recieve.
   * @param type The type of data to send.
   * @param stream The stream to operate on.
   */
  template <typename DType>
  void AllToAllV(
      const DType* const send, const int64_t* send_prefix, DType* const recv,
      const int64_t* recv_prefix, cudaStream_t stream);

  /**
   * @brief Perform an all-to-all with sparse data (idx and value pairs). By
   * necessity, the sizes of each message are variable.
   *
   * @tparam IdType The type of index.
   * @tparam DType The type of value.
   * @param send_idx The set of indexes to send on the device.
   * @param send_value The set of values to send on the device.
   * @param num_feat The number of values per index.
   * @param send_prefix The exclusive prefix sum of elements to send on the
   * host.
   * @param recv_idx The set of indexes to recieve on the device.
   * @param recv_value The set of values to recieve on the device.
   * @param recv_prefix The exclusive prefix sum of the number of elements to
   * recieve on the host.
   * @param stream The stream to communicate on.
   */
  template <typename IdType, typename DType>
  void SparseAllToAll(
      const IdType* send_idx, const DType* send_value, const int64_t num_feat,
      const int64_t* send_prefix, IdType* recv_idx, DType* recv_value,
      const int64_t* recv_prefix, cudaStream_t stream);

  int size() const;

  int rank() const;

  static constexpr const char* _type_key = "cuda.NCCLCommunicator";
  DGL_DECLARE_OBJECT_TYPE_INFO(NCCLCommunicator, Object);

 private:
  ncclComm_t comm_;
  int size_;
  int rank_;
};

DGL_DEFINE_OBJECT_REF(NCCLCommunicatorRef, NCCLCommunicator);

}  // namespace cuda
}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_CUDA_NCCL_API_H_
