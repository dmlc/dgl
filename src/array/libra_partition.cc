/**
 *  Copyright (c) 2021 Intel Corporation
 *
 *  @file distgnn/partition/main_Libra.py
 *  @brief Libra - Vertex-cut based graph partitioner for distirbuted training
 *  @author Vasimuddin Md <vasimuddin.md@intel.com>,
 *          Guixiang Ma <guixiang.ma@intel.com>
 *          Sanchit Misra <sanchit.misra@intel.com>,
 *          Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,
 *          Sasikanth Avancha <sasikanth.avancha@intel.com>
 *          Nesreen K. Ahmed <nesreen.k.ahmed@intel.com>
 */

#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dgl/runtime/parallel_for.h>
#include <dmlc/omp.h>
#include <stdint.h>

#include <vector>

#include "../c_api_common.h"
#include "./check.h"
#include "kernel_decl.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

template <typename IdType>
int32_t Ver2partition(IdType in_val, int64_t *node_map, int32_t num_parts) {
  int32_t pos = 0;
  for (int32_t p = 0; p < num_parts; p++) {
    if (in_val < node_map[p]) return pos;
    pos = pos + 1;
  }
  LOG(FATAL) << "Error: Unexpected output in Ver2partition!";
  return -1;
}

/**
 * @brief Identifies the lead loaded partition/community for a given edge
 * assignment.
 */
int32_t LeastLoad(int64_t *community_edges, int32_t nc) {
  std::vector<int> loc;
  int32_t min = 1e9;
  for (int32_t i = 0; i < nc; i++) {
    if (community_edges[i] < min) {
      min = community_edges[i];
    }
  }
  for (int32_t i = 0; i < nc; i++) {
    if (community_edges[i] == min) {
      loc.push_back(i);
    }
  }

  int32_t r = RandomEngine::ThreadLocal()->RandInt(loc.size());
  CHECK(loc[r] < nc);
  return loc[r];
}

/**
 * @brief Libra - vertexcut based graph partitioning.
 * It takes list of edges from input DGL graph and distributed them among nc
 * partitions During edge distribution, Libra assign a given edge to a partition
 * based on the end vertices, in doing so, it tries to minimized the splitting
 * of the graph vertices. In case of conflict Libra assigns an edge to the least
 * loaded partition/community.
 * @param[in] nc Number of partitions/communities
 * @param[in] node_degree per node degree
 * @param[in] edgenum_unassigned node degree
 * @param[out] community_weights weight of the created partitions
 * @param[in] u src nodes
 * @param[in] v dst nodes
 * @param[out] w weight per edge
 * @param[out] out partition assignment of the edges
 * @param[in] N_n number of nodes in the input graph
 * @param[in] N_e number of edges in the input graph
 * @param[in] prefix output/partition storage location
 */
template <typename IdType, typename IdType2>
void LibraVertexCut(
    int32_t nc, NDArray node_degree, NDArray edgenum_unassigned,
    NDArray community_weights, NDArray u, NDArray v, NDArray w, NDArray out,
    int64_t N_n, int64_t N_e, const std::string &prefix) {
  int32_t *out_ptr = out.Ptr<int32_t>();
  IdType2 *node_degree_ptr = node_degree.Ptr<IdType2>();
  IdType2 *edgenum_unassigned_ptr = edgenum_unassigned.Ptr<IdType2>();
  IdType *u_ptr = u.Ptr<IdType>();
  IdType *v_ptr = v.Ptr<IdType>();
  int64_t *w_ptr = w.Ptr<int64_t>();
  int64_t *community_weights_ptr = community_weights.Ptr<int64_t>();

  std::vector<std::vector<int32_t> > node_assignments(N_n);
  std::vector<IdType2> replication_list;
  // local allocations
  int64_t *community_edges = new int64_t[nc]();
  int64_t *cache = new int64_t[nc]();

  int64_t meter = static_cast<int>(N_e / 100);
  for (int64_t i = 0; i < N_e; i++) {
    IdType u = u_ptr[i];   // edge end vertex 1
    IdType v = v_ptr[i];   // edge end vertex 2
    int64_t w = w_ptr[i];  // edge weight

    CHECK(u < N_n);
    CHECK(v < N_n);

    if (i % meter == 0) {
      fprintf(stderr, ".");
      fflush(0);
    }

    if (node_assignments[u].size() == 0 && node_assignments[v].size() == 0) {
      int32_t c = LeastLoad(community_edges, nc);
      out_ptr[i] = c;
      CHECK_LT(c, nc);

      community_edges[c]++;
      community_weights_ptr[c] = community_weights_ptr[c] + w;
      node_assignments[u].push_back(c);
      if (u != v) node_assignments[v].push_back(c);

      CHECK(node_assignments[u].size() <= static_cast<size_t>(nc))
          << "[bug] 1. generated splits (u) are greater than nc!";
      CHECK(node_assignments[v].size() <= static_cast<size_t>(nc))
          << "[bug] 1. generated splits (v) are greater than nc!";
      edgenum_unassigned_ptr[u]--;
      edgenum_unassigned_ptr[v]--;
    } else if (
        node_assignments[u].size() != 0 && node_assignments[v].size() == 0) {
      for (uint32_t j = 0; j < node_assignments[u].size(); j++) {
        int32_t cind = node_assignments[u][j];
        cache[j] = community_edges[cind];
      }
      int32_t cindex = LeastLoad(cache, node_assignments[u].size());
      int32_t c = node_assignments[u][cindex];
      out_ptr[i] = c;
      community_edges[c]++;
      community_weights_ptr[c] = community_weights_ptr[c] + w;

      node_assignments[v].push_back(c);
      CHECK(node_assignments[v].size() <= static_cast<size_t>(nc))
          << "[bug] 2. generated splits (v) are greater than nc!";
      edgenum_unassigned_ptr[u]--;
      edgenum_unassigned_ptr[v]--;
    } else if (
        node_assignments[v].size() != 0 && node_assignments[u].size() == 0) {
      for (uint32_t j = 0; j < node_assignments[v].size(); j++) {
        int32_t cind = node_assignments[v][j];
        cache[j] = community_edges[cind];
      }
      int32_t cindex = LeastLoad(cache, node_assignments[v].size());
      int32_t c = node_assignments[v][cindex];
      CHECK(c < nc) << "[bug] 2. partition greater than nc !!";
      out_ptr[i] = c;

      community_edges[c]++;
      community_weights_ptr[c] = community_weights_ptr[c] + w;

      node_assignments[u].push_back(c);
      CHECK(node_assignments[u].size() <= static_cast<size_t>(nc))
          << "[bug] 3. generated splits (u) are greater than nc!";
      edgenum_unassigned_ptr[u]--;
      edgenum_unassigned_ptr[v]--;
    } else {
      std::vector<int> setv(nc), intersetv;
      for (int32_t j = 0; j < nc; j++) setv[j] = 0;
      int32_t interset = 0;

      CHECK(node_assignments[u].size() <= static_cast<size_t>(nc))
          << "[bug] 4. generated splits (u) are greater than nc!";
      CHECK(node_assignments[v].size() <= static_cast<size_t>(nc))
          << "[bug] 4. generated splits (v) are greater than nc!";
      for (size_t j = 0; j < node_assignments[v].size(); j++) {
        CHECK(node_assignments[v][j] < nc)
            << "[bug] 4. Part assigned (v) greater than nc!";
        setv[node_assignments[v][j]]++;
      }

      for (size_t j = 0; j < node_assignments[u].size(); j++) {
        CHECK(node_assignments[u][j] < nc)
            << "[bug] 4. Part assigned (u) greater than nc!";
        setv[node_assignments[u][j]]++;
      }

      for (int32_t j = 0; j < nc; j++) {
        CHECK(setv[j] <= 2) << "[bug] 4. unexpected computed value !!!";
        if (setv[j] == 2) {
          interset++;
          intersetv.push_back(j);
        }
      }
      if (interset) {
        for (size_t j = 0; j < intersetv.size(); j++) {
          int32_t cind = intersetv[j];
          cache[j] = community_edges[cind];
        }
        int32_t cindex = LeastLoad(cache, intersetv.size());
        int32_t c = intersetv[cindex];
        CHECK(c < nc) << "[bug] 4. partition greater than nc !!";
        out_ptr[i] = c;
        community_edges[c]++;
        community_weights_ptr[c] = community_weights_ptr[c] + w;
        edgenum_unassigned_ptr[u]--;
        edgenum_unassigned_ptr[v]--;
      } else {
        if (node_degree_ptr[u] < node_degree_ptr[v]) {
          for (uint32_t j = 0; j < node_assignments[u].size(); j++) {
            int32_t cind = node_assignments[u][j];
            cache[j] = community_edges[cind];
          }
          int32_t cindex = LeastLoad(cache, node_assignments[u].size());
          int32_t c = node_assignments[u][cindex];
          CHECK(c < nc) << "[bug] 5. partition greater than nc !!";
          out_ptr[i] = c;
          community_edges[c]++;
          community_weights_ptr[c] = community_weights_ptr[c] + w;

          for (uint32_t j = 0; j < node_assignments[v].size(); j++) {
            CHECK(node_assignments[v][j] != c)
                << "[bug] 5. duplicate partition (v) assignment !!";
          }

          node_assignments[v].push_back(c);
          CHECK(node_assignments[v].size() <= static_cast<size_t>(nc))
              << "[bug] 5. generated splits (v) greater than nc!!";
          replication_list.push_back(v);
          edgenum_unassigned_ptr[u]--;
          edgenum_unassigned_ptr[v]--;
        } else {
          for (uint32_t j = 0; j < node_assignments[v].size(); j++) {
            int32_t cind = node_assignments[v][j];
            cache[j] = community_edges[cind];
          }
          int32_t cindex = LeastLoad(cache, node_assignments[v].size());
          int32_t c = node_assignments[v][cindex];
          CHECK(c < nc) << "[bug] 6. partition greater than nc !!";
          out_ptr[i] = c;
          community_edges[c]++;
          community_weights_ptr[c] = community_weights_ptr[c] + w;
          for (uint32_t j = 0; j < node_assignments[u].size(); j++) {
            CHECK(node_assignments[u][j] != c)
                << "[bug] 6. duplicate partition (u) assignment !!";
          }
          if (u != v) node_assignments[u].push_back(c);

          CHECK(node_assignments[u].size() <= static_cast<size_t>(nc))
              << "[bug] 6. generated splits (u) greater than nc!!";
          replication_list.push_back(u);
          edgenum_unassigned_ptr[u]--;
          edgenum_unassigned_ptr[v]--;
        }
      }
    }
  }
  delete cache;

  for (int64_t c = 0; c < nc; c++) {
    std::string path = prefix + "/community" + std::to_string(c) + ".txt";

    FILE *fp = fopen(path.c_str(), "w");
    CHECK_NE(fp, static_cast<FILE *>(NULL))
        << "Error: can not open file: " << path.c_str();

    for (int64_t i = 0; i < N_e; i++) {
      if (out_ptr[i] == c)
        fprintf(
            fp, "%ld,%ld,%ld\n", static_cast<int64_t>(u_ptr[i]),
            static_cast<int64_t>(v_ptr[i]), w_ptr[i]);
    }
    fclose(fp);
  }

  std::string path = prefix + "/replicationlist.csv";
  FILE *fp = fopen(path.c_str(), "w");
  CHECK_NE(fp, static_cast<FILE *>(NULL))
      << "Error: can not open file: " << path.c_str();

  fprintf(fp, "## The Indices of Nodes that are replicated :: Header");
  printf("\nTotal replication: %ld\n", replication_list.size());

  for (uint64_t i = 0; i < replication_list.size(); i++)
    fprintf(fp, "%ld\n", static_cast<int64_t>(replication_list[i]));

  printf("Community weights:\n");
  for (int64_t c = 0; c < nc; c++) printf("%ld ", community_weights_ptr[c]);
  printf("\n");

  printf("Community edges:\n");
  for (int64_t c = 0; c < nc; c++) printf("%ld ", community_edges[c]);
  printf("\n");

  delete community_edges;
  fclose(fp);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibraVertexCut")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int32_t nc = args[0];
      NDArray node_degree = args[1];
      NDArray edgenum_unassigned = args[2];
      NDArray community_weights = args[3];
      NDArray u = args[4];
      NDArray v = args[5];
      NDArray w = args[6];
      NDArray out = args[7];
      int64_t N = args[8];
      int64_t N_e = args[9];
      std::string prefix = args[10];

      ATEN_ID_TYPE_SWITCH(node_degree->dtype, IdType2, {
        ATEN_ID_TYPE_SWITCH(u->dtype, IdType, {
          LibraVertexCut<IdType, IdType2>(
              nc, node_degree, edgenum_unassigned, community_weights, u, v, w,
              out, N, N_e, prefix);
        });
      });
    });

/**
 * @brief
 * 1. Builds dictionary (ldt) for assigning local node IDs to nodes in the
 *    partitions
 * 2. Builds dictionary (gdt) for storing copies (local ID) of split nodes
 *    These dictionaries will be used in the subsequesnt stages to setup
 *    tracking of split nodes copies across the partition, setting up partition
 *    `ndata` dictionaries.
 * @param[out] a local src node ID of an edge in a partition
 * @param[out] b local dst node ID of an edge in a partition
 * @param[-] indices temporary memory, keeps track of global node ID to local
 *           node ID in a partition
 * @param[out] ldt_key per partition dict for storing global and local node IDs
 *             (consecutive)
 * @param[out] gdt_key global dict for storing number of local nodes (or split
 *             nodes) for a given global node ID
 * @param[out] gdt_value global dict, stores local node IDs (due to split)
 *             across partitions for a given global node ID
 * @param[out] node_map keeps track of range of local node IDs (consecutive)
 *             given to the nodes in the partitions
 * @param[in, out] offset start of the range of local node IDs for this
 *                 partition
 * @param[in] nc number of partitions/communities
 * @param[in] c current partition number
 * @param[in] fsize size of pre-allocated
 *            memory tensor
 * @param[in] prefix input Libra partition file location
 */
List<Value> Libra2dglBuildDict(
    NDArray a, NDArray b, NDArray indices, NDArray ldt_key, NDArray gdt_key,
    NDArray gdt_value, NDArray node_map, NDArray offset, int32_t nc, int32_t c,
    int64_t fsize, const std::string &prefix) {
  int64_t *indices_ptr = indices.Ptr<int64_t>();  // 1D temp array
  int64_t *ldt_key_ptr =
      ldt_key.Ptr<int64_t>();  // 1D local nodes <-> global nodes
  int64_t *gdt_key_ptr = gdt_key.Ptr<int64_t>();  // 1D #split copies per node
  int64_t *gdt_value_ptr = gdt_value.Ptr<int64_t>();  // 2D tensor
  int64_t *node_map_ptr = node_map.Ptr<int64_t>();    // 1D tensor
  int64_t *offset_ptr = offset.Ptr<int64_t>();        // 1D tensor
  int32_t width = nc;

  int64_t *a_ptr = a.Ptr<int64_t>();  // stores local src and dst node ID,
  int64_t *b_ptr = b.Ptr<int64_t>();  // to create the partition graph

  int64_t N_n = indices->shape[0];
  int64_t num_nodes = ldt_key->shape[0];

  for (int64_t i = 0; i < N_n; i++) {
    indices_ptr[i] = -100;
  }

  int64_t pos = 0;
  int64_t edge = 0;
  std::string path = prefix + "/community" + std::to_string(c) + ".txt";
  FILE *fp = fopen(path.c_str(), "r");
  CHECK_NE(fp, static_cast<FILE *>(NULL))
      << "Error: can not open file: " << path.c_str();

  while (!feof(fp) && edge < fsize) {
    int64_t u, v;
    float w;
    CHECK_EQ(
        fscanf(fp, "%ld,%ld,%f\n", &u, &v, &w),
        3);  // reading an edge - the src and dst global node IDs

    if (indices_ptr[u] ==
        -100) {  // if already not assigned a local node ID, local node ID is
      ldt_key_ptr[pos] = u;    // already assigned for this global node ID
      CHECK(pos < num_nodes);  // Sanity check
      indices_ptr[u] =
          pos++;  // consecutive local node ID for a given global node ID
    }
    if (indices_ptr[v] == -100) {  // if already not assigned a local node ID
      ldt_key_ptr[pos] = v;
      CHECK(pos < num_nodes);  // Sanity check
      indices_ptr[v] = pos++;
    }
    a_ptr[edge] = indices_ptr[u];    // new local ID for an edge
    b_ptr[edge++] = indices_ptr[v];  // new local ID for an edge
  }
  CHECK(edge <= fsize)
      << "[Bug] memory allocated for #edges per partition is not enough.";
  fclose(fp);

  List<Value> ret;
  ret.push_back(Value(
      MakeValue(pos)));  // returns total number of nodes in this partition
  ret.push_back(Value(
      MakeValue(edge)));  // returns total number of edges in this partition

  for (int64_t i = 0; i < pos; i++) {
    int64_t u = ldt_key_ptr[i];  // global node ID
    // int64_t  v   = indices_ptr[u];
    int64_t v = i;  // local node ID
    int64_t *ind =
        &gdt_key_ptr[u];  // global dict, total number of local node IDs (an
                          // offset) as of now for a given global node ID
    int64_t *ptr = gdt_value_ptr + u * width;
    ptr[*ind] =
        offset_ptr[0] + v;  // stores a local node ID for the global node ID
    (*ind)++;
    CHECK_NE(v, -100);
    CHECK(*ind <= nc);
  }
  node_map_ptr[c] =
      offset_ptr[0] +
      pos;  // since local node IDs for a partition are consecutive,
            // we maintain the range of local node IDs like this
  offset_ptr[0] += pos;

  return ret;
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglBuildDict")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray a = args[0];
      NDArray b = args[1];
      NDArray indices = args[2];
      NDArray ldt_key = args[3];
      NDArray gdt_key = args[4];
      NDArray gdt_value = args[5];
      NDArray node_map = args[6];
      NDArray offset = args[7];
      int32_t nc = args[8];
      int32_t c = args[9];
      int64_t fsize = args[10];
      std::string prefix = args[11];
      List<Value> ret = Libra2dglBuildDict(
          a, b, indices, ldt_key, gdt_key, gdt_value, node_map, offset, nc, c,
          fsize, prefix);
      *rv = ret;
    });

/**
 * @brief sets up the 1-level tree among the clones of the split-nodes.
 * @param[in] gdt_key global dict for assigning consecutive node IDs to nodes
 *            across all the partitions
 * @param[in] gdt_value global dict for assigning consecutive node IDs to nodes
 *            across all the partition
 * @param[out] lrtensor keeps the root node ID of 1-level tree
 * @param[in] nc number of partitions/communities
 * @param[in] Nn number of nodes in the input graph
 */
void Libra2dglSetLR(
    NDArray gdt_key, NDArray gdt_value, NDArray lrtensor, int32_t nc,
    int64_t Nn) {
  int64_t *gdt_key_ptr = gdt_key.Ptr<int64_t>();      // 1D tensor
  int64_t *gdt_value_ptr = gdt_value.Ptr<int64_t>();  // 2D tensor
  int64_t *lrtensor_ptr = lrtensor.Ptr<int64_t>();    // 1D tensor

  int32_t width = nc;
  int64_t cnt = 0;
  int64_t avg_split_copy = 0, scnt = 0;

  for (int64_t i = 0; i < Nn; i++) {
    if (gdt_key_ptr[i] <= 0) {
      cnt++;
    } else {
      int32_t val = RandomEngine::ThreadLocal()->RandInt(gdt_key_ptr[i]);
      CHECK(val >= 0 && val < gdt_key_ptr[i]);
      CHECK(gdt_key_ptr[i] <= nc);

      int64_t *ptr = gdt_value_ptr + i * width;
      lrtensor_ptr[i] = ptr[val];
    }
    if (gdt_key_ptr[i] > 1) {
      avg_split_copy += gdt_key_ptr[i];
      scnt++;
    }
  }
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglSetLR")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray gdt_key = args[0];
      NDArray gdt_value = args[1];
      NDArray lrtensor = args[2];
      int32_t nc = args[3];
      int64_t Nn = args[4];

      Libra2dglSetLR(gdt_key, gdt_value, lrtensor, nc, Nn);
    });

/**
 * @brief For each node in a partition, it creates a list of remote clone IDs;
 *        also, for each node in a partition, it gathers the data (feats, label,
 *        trian, test) from input graph.
 * @param[out] feat node features in current partition c.
 * @param[in] gfeat input graph node features.
 * @param[out] adj list of node IDs of remote clones.
 * @param[out] inner_nodes marks whether a node is split or not.
 * @param[in] ldt_key per partition dict for tracking global to local node IDs
 * @param[out] gdt_key global dict for storing number of local nodes (or split
 *             nodes) for a given global node ID
 * @param[out] gdt_value global
 *             dict, stores local node IDs (due to split) across partitions for
 *             a given global node ID.
 * @param[in] node_map keeps track of range of local node IDs (consecutive)
 *            given to the nodes in the partitions.
 * @param[out] lr 1-level tree marking for local split nodes.
 * @param[in] lrtensor global (all the partitions) 1-level tree.
 * @param[in] num_nodes number of nodes in current partition.
 * @param[in] nc number of partitions/communities.
 * @param[in] c current partition/community.
 * @param[in] feat_size node feature vector size.
 * @param[out] labels local (for this partition) labels.
 * @param[out] trainm local (for this partition) training nodes.
 * @param[out] testm local (for this partition) testing nodes.
 * @param[out] valm local (for this partition) validation nodes.
 * @param[in] glabels global (input graph) labels.
 * @param[in] gtrainm glabal (input graph) training nodes.
 * @param[in] gtestm glabal (input graph) testing nodes.
 * @param[in] gvalm glabal (input graph) validation nodes.
 * @param[out] Nn number of nodes in the input graph.
 */
template <typename IdType, typename IdType2, typename DType>
void Libra2dglBuildAdjlist(
    NDArray feat, NDArray gfeat, NDArray adj, NDArray inner_node,
    NDArray ldt_key, NDArray gdt_key, NDArray gdt_value, NDArray node_map,
    NDArray lr, NDArray lrtensor, int64_t num_nodes, int32_t nc, int32_t c,
    int32_t feat_size, NDArray labels, NDArray trainm, NDArray testm,
    NDArray valm, NDArray glabels, NDArray gtrainm, NDArray gtestm,
    NDArray gvalm, int64_t Nn) {
  DType *feat_ptr = feat.Ptr<DType>();    // 2D tensor
  DType *gfeat_ptr = gfeat.Ptr<DType>();  // 2D tensor
  int64_t *adj_ptr = adj.Ptr<int64_t>();  // 2D tensor
  int32_t *inner_node_ptr = inner_node.Ptr<int32_t>();
  int64_t *ldt_key_ptr = ldt_key.Ptr<int64_t>();
  int64_t *gdt_key_ptr = gdt_key.Ptr<int64_t>();
  int64_t *gdt_value_ptr = gdt_value.Ptr<int64_t>();  // 2D tensor
  int64_t *node_map_ptr = node_map.Ptr<int64_t>();
  int64_t *lr_ptr = lr.Ptr<int64_t>();
  int64_t *lrtensor_ptr = lrtensor.Ptr<int64_t>();
  int32_t width = nc - 1;

  runtime::parallel_for(0, num_nodes, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      int64_t k = ldt_key_ptr[i];
      int64_t v = i;
      int64_t ind = gdt_key_ptr[k];

      int64_t *adj_ptr_ptr = adj_ptr + v * width;
      if (ind == 1) {
        for (int32_t j = 0; j < width; j++) adj_ptr_ptr[j] = -1;
        inner_node_ptr[i] = 1;
        lr_ptr[i] = -200;
      } else {
        lr_ptr[i] = lrtensor_ptr[k];
        int64_t *ptr = gdt_value_ptr + k * nc;
        int64_t pos = 0;
        CHECK(ind <= nc);
        int32_t flg = 0;
        for (int64_t j = 0; j < ind; j++) {
          if (ptr[j] == lr_ptr[i]) flg = 1;
          if (c != Ver2partition<int64_t>(ptr[j], node_map_ptr, nc))
            adj_ptr_ptr[pos++] = ptr[j];
        }
        CHECK_EQ(flg, 1);
        CHECK(pos == ind - 1);
        for (; pos < width; pos++) adj_ptr_ptr[pos] = -1;
        inner_node_ptr[i] = 0;
      }
    }
  });

  // gather
  runtime::parallel_for(0, num_nodes, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      int64_t k = ldt_key_ptr[i];
      int64_t ind = i * feat_size;
      DType *optr = gfeat_ptr + ind;
      DType *iptr = feat_ptr + k * feat_size;

      for (int32_t j = 0; j < feat_size; j++) optr[j] = iptr[j];
    }

    IdType *labels_ptr = labels.Ptr<IdType>();
    IdType *glabels_ptr = glabels.Ptr<IdType>();
    IdType2 *trainm_ptr = trainm.Ptr<IdType2>();
    IdType2 *gtrainm_ptr = gtrainm.Ptr<IdType2>();
    IdType2 *testm_ptr = testm.Ptr<IdType2>();
    IdType2 *gtestm_ptr = gtestm.Ptr<IdType2>();
    IdType2 *valm_ptr = valm.Ptr<IdType2>();
    IdType2 *gvalm_ptr = gvalm.Ptr<IdType2>();

    for (int64_t i = 0; i < num_nodes; i++) {
      int64_t k = ldt_key_ptr[i];
      CHECK(k >= 0 && k < Nn);
      glabels_ptr[i] = labels_ptr[k];
      gtrainm_ptr[i] = trainm_ptr[k];
      gtestm_ptr[i] = testm_ptr[k];
      gvalm_ptr[i] = valm_ptr[k];
    }
  });
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglBuildAdjlist")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray feat = args[0];
      NDArray gfeat = args[1];
      NDArray adj = args[2];
      NDArray inner_node = args[3];
      NDArray ldt_key = args[4];
      NDArray gdt_key = args[5];
      NDArray gdt_value = args[6];
      NDArray node_map = args[7];
      NDArray lr = args[8];
      NDArray lrtensor = args[9];
      int64_t num_nodes = args[10];
      int32_t nc = args[11];
      int32_t c = args[12];
      int32_t feat_size = args[13];
      NDArray labels = args[14];
      NDArray trainm = args[15];
      NDArray testm = args[16];
      NDArray valm = args[17];
      NDArray glabels = args[18];
      NDArray gtrainm = args[19];
      NDArray gtestm = args[20];
      NDArray gvalm = args[21];
      int64_t Nn = args[22];

      ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Features", {
        ATEN_ID_TYPE_SWITCH(trainm->dtype, IdType2, {
          ATEN_ID_BITS_SWITCH((glabels->dtype).bits, IdType, {
            Libra2dglBuildAdjlist<IdType, IdType2, DType>(
                feat, gfeat, adj, inner_node, ldt_key, gdt_key, gdt_value,
                node_map, lr, lrtensor, num_nodes, nc, c, feat_size, labels,
                trainm, testm, valm, glabels, gtrainm, gtestm, gvalm, Nn);
          });
        });
      });
    });

}  // namespace aten
}  // namespace dgl
