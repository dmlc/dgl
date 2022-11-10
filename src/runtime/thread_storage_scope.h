/**
 *  Copyright (c) 2017 by Contributors
 * @file thread_storage_scope.h
 * @brief Extract thread axis configuration from DGLArgs.
 */
#ifndef DGL_RUNTIME_THREAD_STORAGE_SCOPE_H_
#define DGL_RUNTIME_THREAD_STORAGE_SCOPE_H_

#include <dgl/runtime/packed_func.h>

#include <string>
#include <vector>

namespace dgl {
namespace runtime {

/**
 * @brief Memory hierachy rank in the storage system
 * @note The global rank and shared rank have one to one
 *       correspondence to the thread rank.
 */
enum class StorageRank {
  /** @brief global memory */
  kGlobal = 0,
  /** @brief shared memory among thread group */
  kShared = 1,
  /**
   * @brief reserved for warp memory.
   *  This is only used by programming model.
   *  There is no such memory usually in GPU.
   *  Instead, we can simulate it by registers and shuffle.
   */
  kWarp = 2,
  /** @brief thread local memory */
  kLocal = 3
};

/**
 * @param thread_scope_rank The thread scope rank
 * @return default storage rank given the thread scope
 */
inline StorageRank DefaultStorageRank(int thread_scope_rank) {
  switch (thread_scope_rank) {
    case -1:
      return StorageRank::kGlobal;
    case 0:
      return StorageRank::kShared;
    case 1:
      return StorageRank::kLocal;
    default: {
      LOG(FATAL) << "unknown rank";
      return StorageRank::kGlobal;
    }
  }
}

/** @brief class to represent storage scope */
struct StorageScope {
  /** @brief The rank of the storage */
  StorageRank rank{StorageRank::kGlobal};
  /** @brief tag for special purpose memory. */
  std::string tag;
  // comparator
  inline bool operator==(const StorageScope& other) const {
    return rank == other.rank && tag == other.tag;
  }
  inline bool operator!=(const StorageScope& other) const {
    return !(*this == other);
  }
  inline std::string to_string() const {
    std::string ret;
    switch (rank) {
      case StorageRank::kGlobal:
        return "global" + tag;
      case StorageRank::kShared:
        return "shared" + tag;
      case StorageRank::kWarp:
        return "warp" + tag;
      case StorageRank::kLocal:
        return "local" + tag;
      default:
        LOG(FATAL) << "unknown storage scope";
        return "";
    }
  }
  /**
   * @brief make storage scope from string
   * @param s The string to be parsed.
   * @return The storage scope.
   */
  static StorageScope make(const std::string& s) {
    StorageScope r;
    if (s.compare(0, 6, "global") == 0) {
      r.rank = StorageRank::kGlobal;
      r.tag = s.substr(6, std::string::npos);
    } else if (s.compare(0, 6, "shared") == 0) {
      r.rank = StorageRank::kShared;
      r.tag = s.substr(6, std::string::npos);
    } else if (s.compare(0, 4, "warp") == 0) {
      r.rank = StorageRank::kWarp;
      r.tag = s.substr(4, std::string::npos);
    } else if (s.compare(0, 5, "local") == 0) {
      r.rank = StorageRank::kLocal;
      r.tag = s.substr(5, std::string::npos);
    } else {
      LOG(FATAL) << "unknown storage scope " << s;
    }
    return r;
  }
};

/** @brief class to represent thread scope */
struct ThreadScope {
  /** @brief The rank of thread scope */
  int rank{0};
  /** @brief the dimension index under the rank */
  int dim_index{0};
  /**
   * @brief make storage scope from string
   * @param s The string to be parsed.
   * @return The storage scope.
   */
  static ThreadScope make(const std::string& s) {
    ThreadScope r;
    if (s == "vthread" || s == "cthread") {
      // virtual thread at the same level as local
      r.rank = 1;
      r.dim_index = -1;
    } else if (s.compare(0, 9, "blockIdx.") == 0) {
      r.rank = 0;
      r.dim_index = static_cast<int>(s[9] - 'x');
    } else if (s.compare(0, 10, "threadIdx.") == 0) {
      r.rank = 1;
      r.dim_index = static_cast<int>(s[10] - 'x');
    } else {
      LOG(FATAL) << "Unknown threadscope " << s;
    }
    return r;
  }
};

/** @brief workload speccification */
struct ThreadWorkLoad {
  // array, first three are thread configuration.
  size_t work_size[6];
  /**
   * @param i The block dimension.
   * @return i-th block dim
   */
  inline size_t block_dim(size_t i) const { return work_size[i + 3]; }
  /**
   * @param i The grid dimension.
   * @return i-th grid dim
   */
  inline size_t grid_dim(size_t i) const { return work_size[i]; }
};
/** @brief Thread axis configuration */
class ThreadAxisConfig {
 public:
  void Init(size_t base, const std::vector<std::string>& thread_axis_tags) {
    base_ = base;
    std::vector<bool> filled(6, false);
    for (size_t i = 0; i < thread_axis_tags.size(); ++i) {
      const std::string& tag = thread_axis_tags[i];
      ThreadScope ts = ThreadScope::make(tag);
      arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
      filled[ts.rank * 3 + ts.dim_index] = true;
    }
    work_dim_ = 1;
    for (int i = 0; i < 3; ++i) {
      if (filled[i] || filled[i + 3]) {
        work_dim_ = i + 1;
      }
    }
  }
  // extract workload from arguments.
  ThreadWorkLoad Extract(DGLArgs x) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      w.work_size[arg_index_map_[i]] =
          static_cast<size_t>(x.values[base_ + i].v_int64);
    }
    return w;
  }
  // return the work dim
  size_t work_dim() const { return work_dim_; }

 private:
  /** @brief base axis */
  size_t base_;
  /** @brief The worker dimension */
  size_t work_dim_;
  /** @brief The index mapping. */
  std::vector<uint32_t> arg_index_map_;
};

}  // namespace runtime
}  // namespace dgl

namespace std {
template <>
struct hash<::dgl::runtime::StorageScope> {
  std::size_t operator()(const ::dgl::runtime::StorageScope& k) const {
    return static_cast<size_t>(k.rank);
  }
};
}  // namespace std
#endif  // DGL_RUNTIME_THREAD_STORAGE_SCOPE_H_
