/**
 *  Copyright (c) 2020 by Contributors
 * @file resource_manager.h
 * @brief Manage the resources in the runtime system.
 */
#ifndef DGL_RUNTIME_RESOURCE_MANAGER_H_
#define DGL_RUNTIME_RESOURCE_MANAGER_H_

#include <memory>
#include <string>
#include <unordered_map>

namespace dgl {
namespace runtime {

/**
 * A class that provides the interface to describe a resource that can be
 * managed by a resource manager. Some of the resources cannot be free'd
 * automatically when the process exits, especially when the process doesn't
 * exit normally. One example is shared memory. We can keep track of this kind
 * of resources and manage them properly.
 */
class Resource {
 public:
  virtual ~Resource() {}

  virtual void Destroy() = 0;
};

// Add resource.
void AddResource(const std::string &key, std::shared_ptr<Resource> resource);

// Delete resource.
void DeleteResource(const std::string &key);

// Clean up all resources.
void CleanupResources();

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_RESOURCE_MANAGER_H_
