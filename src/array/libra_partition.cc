/*
Copyright (c) 2021 Intel Corporation
 \file distgnn/partition/main_Libra.py
 \brief Libra - Vertex-cut based graph partitioner for distirbuted training
 \author Vasimuddin Md <vasimuddin.md@intel.com>,
         Guixiang Ma <guixiang.ma@intel.com>
         Sanchit Misra <sanchit.misra@intel.com>,
         Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,         
         Sasikanth Avancha <sasikanth.avancha@intel.com>
         Nesreen K. Ahmed <nesreen.k.ahmed@intel.com>
*/

#include <stdint.h>
#include <omp.h>
#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>
#include <vector>

#ifdef USE_TVM
#include <featgraph.h>
#endif  // USE_TVM

#include "kernel_decl.h"
#include "../c_api_common.h"
#include "./check.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace {

}  // namespace

template<typename IdType>
int32_t Ver2partition(IdType in_val, int32_t* node_map, int32_t num_parts) {
  int32_t pos = 0;
  for (int32_t p=0; p < num_parts; p++) {
    if (in_val < node_map[p])
      return pos;
    pos = pos + 1;
  }
  LOG(FATAL) << "Error: Unexpected output in Ver2partition!";
}

/*! \brief Identifies the lead loaded partition/community for a given edge assignment.*/
int32_t LeastLoad(int64_t* community_edges, int32_t nc) {
  // initialize random seed
  //srand(time(NULL));
  std::vector<int> score, loc;
  int32_t min = 1e9;
  for (int32_t i=0; i < nc; i++) {
    if (community_edges[i] < min) {
      min = community_edges[i];
    }
  }
  for (int32_t i=0; i < nc; i++) {
    if (community_edges[i] == min) {
      loc.push_back(i);
    }
  }
  // int32_t r = rand() % loc.size();   // rand_r recommended
  unsigned int seed = time(NULL);
  int32_t r = rand_r(&seed) % loc.size();   // rand_r recommended
  CHECK(loc[r] < nc);
  return loc[r];
}

/*! \brief Libra - vertexcut based graph partitioning.
  \param nc Number of partitions/communities
  \param node_degree_a Per node degree
  \param edgenum_unassigned_a intermediate memmory as tensor
  \param community_weights_a weight of the created partitions
  \param u_a src nodes
  \param v_a dst nodes
  \param w_a weight per edge
  \param out_a partition assignment of the edges
  \param N_n number of nodes in the input graph
  \param N_e number of edges in the input graph
  \param prefix output/partition storage location
*/
template<typename IdType, typename IdType2, typename DType>
int32_t LibraVertexCut(
  int32_t nc,
  NDArray node_degree_a,
  NDArray edgenum_unassigned_a,
  NDArray community_weights_a,
  NDArray u_a,
  NDArray v_a,
  NDArray w_a,
  NDArray out_a,
  int64_t N_n,
  int64_t N_e,
  std::string prefix) {
  int32_t *out                = out_a.Ptr<int32_t>();
  IdType  *node_degree        = node_degree_a.Ptr<IdType>();
  IdType  *edgenum_unassigned = edgenum_unassigned_a.Ptr<IdType>();
  IdType2 *uptr               = u_a.Ptr<IdType2>();
  IdType2 *vptr               = v_a.Ptr<IdType2>();
  float   *wptr               = w_a.Ptr<float>();
  float   *community_weights  = community_weights_a.Ptr<float>();

  std::vector<std::vector<int32_t> > node_assignments(N_n);
  std::vector<IdType2> replication_list;
  // local allocations
  // int64_t *community_edges = (int64_t*)calloc(sizeof(int64_t), nc);
  // int64_t *cache = (int64_t*)calloc(sizeof(int64_t), nc);
  int64_t *community_edges = reinterpret_cast<int64_t*>(calloc(sizeof(int64_t), nc));
  int64_t *cache = reinterpret_cast<int64_t*>(calloc(sizeof(int64_t), nc));

  CHECK((out_a.GetSize() >> 2) == N_e);
  CHECK((node_degree_a.GetSize() >> 3) == N_n);
  CHECK((u_a.GetSize() >> 3) == N_e);
  CHECK((w_a.GetSize() >> 2) == N_e);

  for (int64_t i=0; i < N_e; i++) {
    IdType u = uptr[i];
    IdType v = vptr[i];
    float  w = wptr[i];
    CHECK(u < N_n);
    CHECK(v < N_n);

    if (i%10000000 == 0) {
      printf("."); fflush(0);
    }

    if (node_assignments[u].size() == 0 && node_assignments[v].size() == 0) {
      int32_t c = LeastLoad(community_edges, nc);
      out[i] = c;
      CHECK_LT(c, nc);

      community_edges[c]++;
      community_weights[c] = community_weights[c] + w;
      node_assignments[u].push_back(c);
      if (u != v)
        node_assignments[v].push_back(c);

      CHECK(node_assignments[u].size() <= nc) <<
        "Error: 1. generated splits (u) are greater than nc!";
      CHECK(node_assignments[v].size() <= nc) <<
        "Error: 1. generated splits (v) are greater than nc!";
      edgenum_unassigned[u]--;
      edgenum_unassigned[v]--;
    } else if (node_assignments[u].size() != 0 && node_assignments[v].size() == 0) {
      for (uint32_t j=0; j < node_assignments[u].size(); j++) {
        int32_t cind = node_assignments[u][j];
        cache[j] = community_edges[cind];
      }
      int32_t cindex = LeastLoad(cache, node_assignments[u].size());
      int32_t c = node_assignments[u][cindex];
      out[i] = c;
      community_edges[c]++;
      community_weights[c] = community_weights[c] + w;

      node_assignments[v].push_back(c);
      CHECK(node_assignments[v].size() <= nc) <<
        "Error: 2. generated splits (v) are greater than nc!";
      edgenum_unassigned[u]--;
      edgenum_unassigned[v]--;
    } else if (node_assignments[v].size() != 0 && node_assignments[u].size() == 0) {
      for (uint32_t j=0; j < node_assignments[v].size(); j++) {
        int32_t cind = node_assignments[v][j];
        cache[j] = community_edges[cind];
      }
      int32_t cindex = LeastLoad(cache, node_assignments[v].size());
      int32_t c = node_assignments[v][cindex];
      CHECK(c < nc) << "Error: 2. partition greater than nc !!";
      out[i] = c;

      community_edges[c]++;
      community_weights[c] = community_weights[c] + w;

      node_assignments[u].push_back(c);
      CHECK(node_assignments[u].size() <= nc) <<
        "3. Error: generated splits (u) are greater than nc!";
      edgenum_unassigned[u]--;
      edgenum_unassigned[v]--;
    } else {
      std::vector<int> setv(nc), intersetv;
      for (int32_t j=0; j < nc; j++) setv[j] = 0;
      int32_t interset = 0;

      CHECK(node_assignments[u].size() <= nc) <<
        "4. Error: generated splits (u) are greater than nc!";
      CHECK(node_assignments[v].size() <= nc) <<
        "4. Error: generated splits (v) are greater than nc!";
      for (int32_t j=0; j < node_assignments[v].size(); j++) {
        CHECK(node_assignments[v][j] < nc) << "Error: 4. Part assigned (v) greater than nc!";
        setv[node_assignments[v][j]]++;
      }

      for (int32_t j=0; j < node_assignments[u].size(); j++) {
        CHECK(node_assignments[u][j] < nc) << "Error: 4. Part assigned (u) greater than nc!";
        setv[node_assignments[u][j]]++;
      }

      for (int32_t j=0; j < nc; j++) {
        CHECK(setv[j] <= 2) << "Error: 4. unexpected computed value !!!";
        if (setv[j] == 2) {
          interset++;
          intersetv.push_back(j);
        }
      }
      if (interset) {
        for (int32_t j=0; j < intersetv.size(); j++) {
          int32_t cind = intersetv[j];
          cache[j] = community_edges[cind];
        }
        int32_t cindex = LeastLoad(cache, intersetv.size());
        int32_t c = intersetv[cindex];        
        CHECK(c < nc) << "Error: 4. partition greater than nc !!";
        out[i] = c;
        community_edges[c]++;
        community_weights[c] = community_weights[c] + w;
        edgenum_unassigned[u]--;
        edgenum_unassigned[v]--;
      } else {
        if (node_degree[u] < node_degree[v]) {
          for (uint32_t j=0; j < node_assignments[u].size(); j++) {
            int32_t cind = node_assignments[u][j];
            cache[j] = community_edges[cind];
          }
          int32_t cindex = LeastLoad(cache, node_assignments[u].size());
          int32_t c = node_assignments[u][cindex];
          CHECK(c < nc) << "Error: 5. partition greater than nc !!";
          out[i] = c;
          community_edges[c]++;
          community_weights[c] = community_weights[c] + w;

          for (uint32_t j=0; j < node_assignments[v].size(); j++) {
            CHECK(node_assignments[v][j] != c) <<
              "Error: 5. duplicate partition (v) assignment !!";
          }

          node_assignments[v].push_back(c);
          CHECK(node_assignments[v].size() <= nc) <<
            "Error: 5. generated splits (v) greater than nc!!";
          replication_list.push_back(v);
          edgenum_unassigned[u]--;
          edgenum_unassigned[v]--;
        } else {
          for (uint32_t j=0; j < node_assignments[v].size(); j++) {
            int32_t cind = node_assignments[v][j];
            cache[j] = community_edges[cind];
          }
          int32_t cindex = LeastLoad(cache, node_assignments[v].size());
          int32_t c = node_assignments[v][cindex];
          CHECK(c < nc)  << "Error: 6. partition greater than nc !!";
          out[i] = c;
          community_edges[c]++;
          community_weights[c] = community_weights[c] + w;
          for (uint32_t j=0; j < node_assignments[u].size(); j++) {
            CHECK(node_assignments[u][j] != c) <<
              "Error: 6. duplicate partition (u) assignment !!";
          }
          if (u != v)
            node_assignments[u].push_back(c);

          CHECK(node_assignments[u].size() <= nc) <<
            "Error: 6. generated splits (u) greater than nc!!";
          replication_list.push_back(u);
          edgenum_unassigned[u]--;
          edgenum_unassigned[v]--;
        }
      }
    }
  }
  free(cache);

  std::string lprefix = prefix;
  for (int64_t c=0; c < nc; c++) {
    std::string str = lprefix + "/" + std::to_string(nc) + "Communities/community" +
      std::to_string(c) +".txt";

    FILE *fp = fopen(str.c_str(), "w");
    CHECK_NOTNULL(fp);

    for (int64_t i=0; i < N_e; i++) {
      if (out[i] == c)
        fprintf(fp, "%ld,%ld,%f\n", uptr[i], vptr[i], wptr[i]);
    }
    fclose(fp);
  }

  std::string str = lprefix + "/"  + std::to_string(nc) + "Communities/replicationlist.csv";
  FILE *fp = fopen(str.c_str(), "w");
  CHECK_NOTNULL(fp);

  std::string str_ = "# The Indices of Nodes that are replicated :: Header";
  fprintf(fp, "## The Indices of Nodes that are replicated :: Header");
  printf("Total replication: %ld\n", replication_list.size());

  for (uint64_t i=0; i < replication_list.size(); i++)
    fprintf(fp, "%ld\n", replication_list[i]);

  printf("Community weights:\n");
  for (int64_t c=0; c < nc; c++)
    printf("%f ", community_weights[c]);
  printf("\n");

  printf("Community edges:\n");
  for (int64_t c=0; c < nc; c++)
    printf("%ld ", community_edges[c]);
  printf("\n");

  free(community_edges);
  fclose(fp);

  return 0;
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibraVertexCut")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int32_t nc                 = args[0];
  NDArray node_degree        = args[1];
  NDArray edgenum_unassigned = args[2];
  NDArray community_weights  = args[3];
  NDArray u                  = args[4];
  NDArray v                  = args[5];
  NDArray w                  = args[6];
  NDArray out                = args[7];
  int64_t N                  = args[8];
  int64_t N_e                = args[9];
  std::string prefix        = args[10];

  ATEN_ID_TYPE_SWITCH(node_degree->dtype, IdType2, {
      ATEN_ID_TYPE_SWITCH(u->dtype, IdType, {
          ATEN_FLOAT_TYPE_SWITCH(w->dtype, DType, "Feature data", {
              LibraVertexCut<IdType, IdType2, DType>(nc,
                                                     node_degree,
                                                     edgenum_unassigned,
                                                     community_weights,
                                                     u,
                                                     v,
                                                     w,
                                                     out,
                                                     N,
                                                     N_e,
                                                     prefix);
            });
        });
    });
});


/*! \brief Builds dictionaries for assigning local node IDs to nodes in the paritions
  and prepares indices to gather the graph from input graph.
  \param a_a local src node ID of an edge in a partition
  \param b_a local dst node ID of an edge in a partition
  \param indices_a keep track of global node ID to local node ID
  \param ldt_key_a per partition dict for tacking gloal to local node IDs
  \param gdt_key_a global dict for assigning consecutive node IDs to nodes across all the
  partitions
  \param gdt_value_a global dict for assigning consecutive node IDs to nodes across all the
  partition
  \param node_map_a keeps track of range of local node IDs (consecutive) given to the nodes in
  the partitions
  \param nc  number of partitions/communities
  \param c current partition numbers
  \param fsize size of pre-allocated memory tensor
  \param hash_nodes_a returns the actual number of nodes and edges in the partition
  \param prefix output file location
 */
void Libra2dglBuildDict(
  NDArray a_a,
  NDArray b_a,
  NDArray indices_a,
  NDArray ldt_key_a,
  NDArray gdt_key_a,
  NDArray gdt_value_a,
  NDArray node_map_a,
  NDArray offset_a,
  int32_t nc,
  int32_t c,
  int64_t fsize,
  NDArray hash_nodes_a,
  std::string prefix) {
  int32_t *indices    = indices_a.Ptr<int32_t>();
  int64_t *ldt_key    = ldt_key_a.Ptr<int64_t>();
  int32_t *gdt_key    = gdt_key_a.Ptr<int32_t>();
  int32_t *gdt_value  = gdt_value_a.Ptr<int32_t>();   // 2D tensor
  int32_t *node_map   = node_map_a.Ptr<int32_t>();
  int32_t *offset     = offset_a.Ptr<int32_t>();
  int32_t *hash_nodes = hash_nodes_a.Ptr<int32_t>();
  int32_t width = nc;

  int64_t *a = a_a.Ptr<int64_t>();
  int64_t *b = b_a.Ptr<int64_t>();

  int32_t N_n = indices_a.GetSize() >> 2;
  int32_t num_nodes = ldt_key_a.GetSize() >> 2;

  #pragma omp simd
  for (int32_t i=0; i < N_n; i++) {
    indices[i] = -100;
  }

  int32_t pos = 0;
  int64_t edge = 0;
  std::string lprefix = prefix;
  std::string str = lprefix + "/community" + std::to_string(c) + ".txt";
  FILE *fp = fopen(str.c_str(), "r");
  CHECK_NOTNULL(fp);

  while (!feof(fp) && edge < fsize) {
    int64_t u, v;
    float w;
    fscanf(fp, "%ld,%ld,%f\n", &u, &v, &w);

    if (indices[u] == -100) {
      ldt_key[pos] = u;
      CHECK(pos < num_nodes);
      indices[u] = pos++;
    }
    if (indices[v] == -100) {
      ldt_key[pos] = v;
      CHECK(pos < num_nodes);
      indices[v] = pos++;
    }
    a[edge] = indices[u];
    b[edge++] = indices[v];
  }
  CHECK(edge <= fsize);
  fclose(fp);
  hash_nodes[0] = pos;
  hash_nodes[1] = edge;

  for (int32_t i=0; i < pos; i++) {
    int64_t  u   = ldt_key[i];
    int32_t  v   = indices[u];
    int32_t *ind = &gdt_key[u];
    int32_t *ptr = gdt_value + u*width;
    ptr[*ind] = offset[0] + v;
    (*ind)++;
    // CHECK(v != -100);
    CHECK_NE(v, -100);
    CHECK(*ind <= nc);
  }
  node_map[c] = offset[0] +  pos;
  offset[0] += pos;
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglBuildDict")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray     a          = args[0];
  NDArray     b          = args[1];
  NDArray     indices    = args[2];
  NDArray     ldt_key    = args[3];
  NDArray     gdt_key    = args[4];
  NDArray     gdt_value  = args[5];
  NDArray     node_map   = args[6];
  NDArray     offset     = args[7];
  int32_t     nc         = args[8];
  int32_t     c          = args[9];
  int64_t     fsize      = args[10];
  NDArray     hash_nodes = args[11];
  std::string prefix     = args[12];
  Libra2dglBuildDict(a, b, indices, ldt_key, gdt_key,
                     gdt_value, node_map, offset,
                     nc, c, fsize, hash_nodes, prefix);
});


/*! \brief sets up the 1-level tree aomng the clones of the split-nodes.
  \param gdt_key_a global dict for assigning consecutive node IDs to nodes across all the
  partitions
  \param gdt_value_a global dict for assigning consecutive node IDs to nodes across all the
  partition
  \param lftensor_a keeps the root node ID of 1-level tree
  \param nc number of partitions/communities
  \param Nn number of nodes in the input graph
 */
void Libra2dglSetLF(
  NDArray gdt_key_a,
  NDArray gdt_value_a,
  NDArray lftensor_a,
  int32_t nc,
  int64_t Nn) {
  srand(time(NULL));
  int32_t *gdt_key    = gdt_key_a.Ptr<int32_t>();
  int32_t *gdt_value  = gdt_value_a.Ptr<int32_t>();   // 2D tensor
  int32_t *lftensor   = lftensor_a.Ptr<int32_t>();

  int32_t width = nc;
  int32_t cnt = 0;
  int32_t avg_split_copy = 0, scnt = 0;

  for (int64_t i=0; i < Nn; i++) {
    if (gdt_key[i] <= 0) {
      cnt++;
    } else {
      unsigned int seed = time(NULL);
      int32_t val = rand_r(&seed) % gdt_key[i];  // consider using rand_r()
      CHECK(val >= 0 && val < gdt_key[i]);
      CHECK(gdt_key[i] <= nc);

      int32_t *ptr = gdt_value + i*width;
      lftensor[i] = ptr[val];
    }
    if (gdt_key[i] > 1) {
      avg_split_copy += gdt_key[i];
      scnt++;
    }
  }
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglSetLF")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray gdt_key   = args[0];
  NDArray gdt_value = args[1];
  NDArray lftensor  = args[2];
  int32_t nc        = args[3];
  int64_t Nn        = args[4];

  Libra2dglSetLF(gdt_key, gdt_value, lftensor, nc, Nn);
});


/*!
  \brief For each node in a partition, it creates a list of remote clone IDs;
  also, for each node in a partition, it gathers the data (feats, label, trian, test)
  from input graph.
  \param feat_a node features in current partition c
  \param gfeat_a input graph node features
  \param adj_a list of node IDs of remote clones
  \param inner_nodes_a marks whether a node is split or not
  \param ldt_key_a per partition dict for tacking gloal to local node IDs
  \param gdt_key_a global dict for assigning consecutive node IDs to nodes across all the
  partitions
  \param gdt_value_a global dict for assigning consecutive node IDs to nodes across all the
  partition
  \param node_map_a keeps track of range of local node IDs (consecutive) given to the nodes in
  the partitions
  \param lf_a 1-level tree marking for local split nodes
  \param lftensor_a gloabl (all the partitions) 1-level tree
  \param num_nodes number of nodes in current partition
  \param nc number of partitions/communities
  \param c current partition/community
  \param feat_size node feature vector size
  \param labels_a local (for this partition) labels
  \param trainm_a local (for this partition) training nodes
  \param testm_a local (for this partition) testing nodes
  \param valm_a local (for this partition) validation nodes
  \param glabels_a gloabl (input graph) labels
  \param gtrainm_a gloabl (input graph) training nodes
  \param gtestm_a gloabl (input graph) testing nodes
  \param gvalm_a gloabl (input graph) validation nodes
  \param Nn number of nodes in the input graph
 */
template<typename IdType, typename DType>
void Libra2dglBuildAdjlist(
  NDArray feat_a,
  NDArray gfeat_a,
  NDArray adj_a,
  NDArray inner_node_a,
  NDArray ldt_key_a,
  NDArray gdt_key_a,
  NDArray gdt_value_a,
  NDArray node_map_a,
  NDArray lf_a,
  NDArray lftensor_a,
  int32_t num_nodes,
  int32_t nc,
  int32_t c,
  int32_t feat_size,
  NDArray labels_a ,
  NDArray trainm_a ,
  NDArray testm_a  ,
  NDArray valm_a   ,
  NDArray glabels_a,
  NDArray gtrainm_a,
  NDArray gtestm_a ,
  NDArray gvalm_a,
  int64_t Nn) {
  DType   *feat       = feat_a.Ptr<DType>();    // 2D tensor
  DType   *gfeat      = gfeat_a.Ptr<DType>();   // 2D tensor
  int32_t *adj        = adj_a.Ptr<int32_t>();   // 2D tensor
  int32_t *inner_node = inner_node_a.Ptr<int32_t>();
  int64_t *ldt_key    = ldt_key_a.Ptr<int64_t>();
  int32_t *gdt_key    = gdt_key_a.Ptr<int32_t>();
  int32_t *gdt_value  = gdt_value_a.Ptr<int32_t>();   // 2D tensor
  int32_t *node_map   = node_map_a.Ptr<int32_t>();
  int32_t *lf         = lf_a.Ptr<int32_t>();
  int32_t *lftensor   = lftensor_a.Ptr<int32_t>();
  int32_t width = nc - 1;

  #pragma omp parallel for
  for (int32_t i=0; i < num_nodes; i++) {
    int64_t k = ldt_key[i];
    int64_t v = i;
    int32_t ind = gdt_key[k];

    int32_t *adj_ptr = adj + v*width;
    if (ind == 1) {
      for (int32_t j=0; j < width; j++) adj_ptr[j] = -1;
      inner_node[i] = 1;
      lf[i] = -200;
    } else {
      lf[i] = lftensor[k];
      int32_t *ptr = gdt_value + k*nc;
      int32_t pos = 0;
      CHECK(ind <= nc);
      int32_t flg = 0;
      for (int32_t j=0; j < ind; j++) {
        if (ptr[j] == lf[i]) flg = 1;
        if (c != Ver2partition<int32_t>(ptr[j], node_map, nc) )
          adj_ptr[pos++] = ptr[j];
      }
      CHECK_EQ(flg, 1);
      CHECK(pos == ind - 1);
      for (; pos < width; pos++) adj_ptr[pos] = -1;
      inner_node[i] = 0;
    }
  }
  // gathering
  #pragma omp parallel for
  for (int64_t i=0; i < num_nodes; i++) {
    int64_t k = ldt_key[i];
    int64_t ind = i*feat_size;
    DType *optr = gfeat + ind;
    DType *iptr = feat + k*feat_size;

    for (int32_t j=0; j < feat_size; j++)
      optr[j] = iptr[j];
  }

  IdType *labels = labels_a.Ptr<IdType>();
  IdType *glabels = glabels_a.Ptr<IdType>();
  bool *trainm = trainm_a.Ptr<bool>();
  bool *gtrainm = gtrainm_a.Ptr<bool>();
  bool *testm = testm_a.Ptr<bool>();
  bool *gtestm = gtestm_a.Ptr<bool>();
  bool *valm = valm_a.Ptr<bool>();
  bool *gvalm = gvalm_a.Ptr<bool>();
  #pragma omp parallel for
  for (int64_t i=0; i < num_nodes; i++) {
    int64_t k = ldt_key[i];
    CHECK(k >=0 && k < Nn);
    glabels[i] = labels[k];
    gtrainm[i] = trainm[k];
    gtestm[i] = testm[k];
    gvalm[i] = valm[k];
  }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglBuildAdjlist")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray feat       = args[0];
  NDArray gfeat      = args[1];
  NDArray adj        = args[2];
  NDArray inner_node = args[3];
  NDArray ldt_key    = args[4];
  NDArray gdt_key    = args[5];
  NDArray gdt_value  = args[6];
  NDArray node_map   = args[7];
  NDArray lf         = args[8];
  NDArray lftensor   = args[9];
  int32_t num_nodes  = args[10];
  int32_t nc         = args[11];
  int32_t c          = args[12];
  int32_t feat_size  = args[13];
  NDArray labels     = args[14];
  NDArray trainm     = args[15];
  NDArray testm      = args[16];
  NDArray valm       = args[17];
  NDArray glabels    = args[18];
  NDArray gtrainm    = args[19];
  NDArray gtestm     = args[20];
  NDArray gvalm      = args[21];
  int64_t Nn         = args[22];

  ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Features", {
      ATEN_ID_BITS_SWITCH((glabels->dtype).bits, IdType, {
          Libra2dglBuildAdjlist<IdType, DType>(feat, gfeat,
                                               adj, inner_node,
                                               ldt_key, gdt_key,
                                               gdt_value,
                                               node_map, lf,
                                               lftensor, num_nodes,
                                               nc, c,
                                               feat_size, labels,
                                               trainm, testm,
                                               valm, glabels,
                                               gtrainm, gtestm,
                                               gvalm, Nn);
        });
    });
});


}  // namespace aten
}  // namespace dgl
