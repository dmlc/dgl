/**
 *  Copyright (c) 2020 by Contributors
 *
 * Copied from dgl/src/runtime/resource_manager.cc. Modifications to one of
 * these files should be propagated to the other.
 *
 * @file shm/resource_manager.cc
 * @brief Manage the resources.
 */
#include "./resource_manager.h"

#include <torch/torch.h>

#include <utility>

namespace graphbolt {
namespace sampling {

/**
 * The runtime allocates resources during the computation. Some of the resources
 * cannot be destroyed after the process exits especially when the process
 * doesn't exits normally. We need to keep track of the resources in the system
 * and clean them up properly.
 */
class ResourceManager {
  std::unordered_map<std::string, std::shared_ptr<Resource>> resources;

 public:
  void Add(const std::string &key, std::shared_ptr<Resource> resource) {
    auto it = resources.find(key);
    TORCH_CHECK(it == resources.end(), key, " already exists");
    resources.insert(
        std::pair<std::string, std::shared_ptr<Resource>>(key, resource));
  }

  void Erase(const std::string &key) { resources.erase(key); }

  void Cleanup() {
    for (auto it = resources.begin(); it != resources.end(); it++) {
      it->second->Destroy();
    }
    resources.clear();
  }
};

static ResourceManager manager;

void AddResource(const std::string &key, std::shared_ptr<Resource> resource) {
  manager.Add(key, resource);
}

void DeleteResource(const std::string &key) { manager.Erase(key); }

void CleanupResources() { manager.Cleanup(); }

}  // namespace sampling
}  // namespace graphbolt
