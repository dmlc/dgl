/**
 *  Copyright (c) 2020 by Contributors
 * @file rpc/server_state.h
 * @brief Implementation of RPC utilities used by both server and client sides.
 */

#ifndef DGL_RPC_SERVER_STATE_H_
#define DGL_RPC_SERVER_STATE_H_

#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>

#include <string>
#include <unordered_map>

namespace dgl {
namespace rpc {

/**
 * @brief Data stored in one DGL server.
 *
 * In a distributed setting, DGL partitions all data associated with the graph
 * (e.g., node and edge features, graph structure, etc.) to multiple partitions,
 * each handled by one DGL server. Hence, the ServerState class includes all
 * the data associated with a graph partition.
 *
 * Under some setup, users may want to deploy servers in a heterogeneous way
 * -- servers are further divided into special groups for fetching/updating
 * node/edge data and for sampling/querying on graph structure respectively.
 * In this case, the ServerState can be configured to include only node/edge
 * data or graph structure.
 *
 * Each machine can have multiple server and client processes, but only one
 * server is the *master* server while all the others are backup servers. All
 * clients and backup servers share the state of the master server via shared
 * memory, which means the ServerState class must be serializable and large
 * bulk data (e.g., node/edge features) must be stored in NDArray to leverage
 * shared memory.
 */
struct ServerState : public runtime::Object {
  /** @brief Key value store for NDArray data */
  std::unordered_map<std::string, runtime::NDArray> kv_store;

  /** @brief Graph structure of one partition */
  HeteroGraphPtr graph;

  /** @brief Total number of nodes */
  int64_t total_num_nodes = 0;

  /** @brief Total number of edges */
  int64_t total_num_edges = 0;

  static constexpr const char* _type_key = "server_state.ServerState";
  DGL_DECLARE_OBJECT_TYPE_INFO(ServerState, runtime::Object);
};
DGL_DEFINE_OBJECT_REF(ServerStateRef, ServerState);

}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_SERVER_STATE_H_
