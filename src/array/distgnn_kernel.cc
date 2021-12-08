/*
Copyright (c) 2021 Intel Corporation
 \file distgnn/partition/main_Libra.py
 \brief Libra - Vertex-cut based graph partitioner for distributed training
 \author Vasimuddin Md <vasimuddin.md@intel.com>,
         Sanchit Misra <sanchit.misra@intel.com>,
         Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,
         Sasikanth Avancha <sasikanth.avancha@intel.com>
*/

#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>
#include <vector>
#include <stdint.h>
#include <omp.h>

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

/* DistGNN functions */
template<typename IdType>
int32_t Ver2partition_(IdType in_val, int32_t* node_map, int32_t num_parts) {
  int32_t pos = 0;
  for (int32_t p=0; p<num_parts; p++) {
    if (in_val < node_map[p])
      return pos;
    pos = pos + 1;
  }
  LOG(FATAL) << "Error: Unexpected output in Ver2partition_!";
}


int32_t Map2localIndex(int32_t otf, int32_t *node_map,
                       int32_t num_parts, int32_t &pid,
                       int32_t cur_part) {
  int32_t last_p = 0;
  if (cur_part != 0)
    last_p = node_map[cur_part - 1];
  for (int32_t i=cur_part; i<num_parts; i++) {
    int32_t nnodes = node_map[i];
    if (otf < nnodes) {
      pid = i;
      return otf - last_p;
    }
    last_p = nnodes;
  }
  LOG(FATAL) << "Error: Unexpected output in Map2localIndex!";
}


/*!
  \brief Leaf nodes gather local node aggregates to send to remote clone root node.
  \param feat_a node features in current partition
  \param Nn number of nodes in the input graph
  \param adj_a list of node IDs of remote clones
  \param send_feat_list_a buffer for gathered aggregates
  \param send_to_node_list_a remote node IDs storage for use during root to leaf comm
  \param selected_nodes_a split nodes in current partition
  \param in_degs_a node degree
  \param ver2part_a remote clone node's partition
  \param ver2part_index_a remote clone node's local ID
  \param width total possible clone nodes per node
  \param feat_size feature vector size
  \param cur_part remote partition/rank for which this gather op is invoked
  \param soffset_base used a offset due to split in communicaiton
  \param soffset_cur used a offset due to split in communicaiton
  \param node_map_a node ID range for all the partitions
  \param num_parts number of mpi ranks or partitions
*/
template<typename IdType, typename DType>
void FdrpaGatherEmbLR(
  NDArray feat_a,
  int64_t Nn,
  NDArray adj_a,
  NDArray send_feat_list_a,
  int64_t offset,
  NDArray send_node_list_a,
  NDArray send_to_node_list_a,
  NDArray selected_nodes_a,
  NDArray in_degs_a,
  NDArray ver2part_a,
  NDArray ver2part_index_a,
  int32_t width,
  int32_t feat_size,
  int32_t cur_part,
  int64_t soffset_base,
  int64_t soffset_cur,
  NDArray node_map_a,
  int32_t num_parts) {
  
  if (soffset_cur == 0) return;
  
  DType*   feat              = feat_a.Ptr<DType>();
  IdType*  adj               = adj_a.Ptr<IdType>();
  DType*   send_feat_list    = send_feat_list_a.Ptr<DType>() + offset * (feat_size + 1);
  int32_t* send_node_list    = send_node_list_a.Ptr<int32_t>();
  int32_t* send_to_node_list = send_to_node_list_a.Ptr<int32_t>() + offset;
  int32_t* selected_nodes    = selected_nodes_a.Ptr<int32_t>();
  DType*   in_degs           = in_degs_a.Ptr<DType>();
  int32_t* ver2part          = ver2part_a.Ptr<int32_t>();
  int32_t* ver2part_index    = ver2part_index_a.Ptr<int32_t>();
  int32_t* node_map          = node_map_a.Ptr<int32_t>();
  
  int32_t pos = 0, pos_out = 0;
  int32_t len = selected_nodes_a.GetSize() >> 2;
  
  int32_t *gindex = (int32_t*)calloc(sizeof(int32_t), len);
  int32_t lim = send_node_list_a.GetSize() >> 2;
  int32_t counter = 0;
  pos = 0;
  
  for (int32_t i=0; i<len; i++) {
    if (ver2part[i] == cur_part) {
      if (counter >= soffset_base && pos < soffset_cur) {
        gindex[pos++] = i;
      }
      counter ++;
    }
  }
  CHECK(pos == soffset_cur);
  
  #pragma omp parallel for
  for (int64_t i=0; i<pos; i++) {
    int32_t id = gindex[i];
    int64_t index = selected_nodes[id];
    
    #if DEBUG  //checks
    CHECK(index >= 0 && index < Nn);
    #endif
    
    DType *iptr = feat + index * feat_size;
    send_node_list[i] = index;
    DType *optr = send_feat_list + i * (feat_size + 1);
    optr[0] = in_degs[index];
    send_to_node_list[i] = ver2part_index[id];
    
    #if DEBUG   // checks
    int32_t  p = Ver2partition_<int32_t>(ver2part_index[id], node_map, num_parts);
    if (cur_part != p)
      LOG(FATAL) << "Error: Cur_part not matching !";
    CHECK(cur_part == p);
    #endif
    
    #pragma omp simd
    for (int64_t k=0; k<feat_size; k++)
      optr[1 + k] = iptr[k];
  }
  free(gindex);
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaGatherEmbLR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray feat              = args[0];
  int64_t Nn                = args[1];
  NDArray adj               = args[2];
  NDArray send_feat_list    = args[3];
  int64_t offset            = args[4];
  NDArray send_node_list    = args[5];
  NDArray send_to_node_list = args[6];
  NDArray selected_nodes    = args[7];
  NDArray in_degs           = args[8];
  NDArray ver2part          = args[9];
  NDArray ver2part_index    = args[10];
  int32_t width             = args[11];
  int32_t feat_size         = args[12];
  int32_t cur_part          = args[13];
  int64_t soffset_base      = args[14];
  int64_t soffset_cur       = args[15];
  NDArray node_map          = args[16];
  int32_t num_parts         = args[17];

  ATEN_ID_TYPE_SWITCH(adj->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Feature data", {
          FdrpaGatherEmbLR<IdType, DType>(feat, Nn,
                                          adj, send_feat_list,
                                          offset, send_node_list,
                                          send_to_node_list, selected_nodes,
                                          in_degs, ver2part,
                                          ver2part_index, width,
                                          feat_size, cur_part,
                                          soffset_base, soffset_cur,
                                          node_map, num_parts);
        });
    });
});


/*!
  \brief Root node scatter reduces the recvd partial aggregates to local partial aggregates.
  \param otf_a recvd partial aggreagated from leafs
  \param offsetf offset in otf_a
  \param otn_a local node IDs for which partial aggregates are recvd
  \param offsetn offset in otn_a
  \param feat_a node features in current partition
  \param in_degs_a node degree
  \param node_map_a node ID range for all the partitions
  \param dim recvd partial aggregate size
  \param feat_size node feature vector size
  \param num_parts number of mpi ranks or partitions
  \param recv_list_nodes_a buffer to store recv nodes from individual partitions,used in next gather
  \param pos debug variable
  \param lim number of nodes recvd from a remote parition/mpi rank
  \param cur_part my mpi rank
*/
template<typename IdType, typename DType>
void ScatterReduceLR(
  NDArray otf_a,
  int64_t offsetf,
  NDArray otn_a,
  int64_t offsetn,
  NDArray feat_a,
  NDArray in_degs_a,
  NDArray node_map_a,
  int64_t dim,
  int64_t feat_size,
  int64_t num_parts,
  NDArray recv_list_nodes_a,
  NDArray pos_a,
  int64_t lim,
  int32_t cur_part) {
  
  DType*   otf             = otf_a.Ptr<DType>() + offsetf;
  int32_t* otn             = otn_a.Ptr<int32_t>() + offsetn;
  DType*   feat            = feat_a.Ptr<DType>();
  DType*  in_degs          = in_degs_a.Ptr<DType>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  int32_t* recv_list_nodes = recv_list_nodes_a.Ptr<int32_t>();
  int64_t* pos             = pos_a.Ptr<int64_t>();

  int64_t size = in_degs_a.GetSize() >> 2;
  int64_t fsize = feat_a.GetSize() >> 2;

  #pragma omp parallel for
  for (int64_t i=0; i<lim; i++) {
    DType *iptr = otf + i * (feat_size + 1);
    CHECK(iptr[0] >= 0) << "Error: ScatterReduceLR, -ve index!";

    int32_t pid = -1;
    int64_t index = Map2localIndex(otn[i], node_map, num_parts, pid, cur_part);
    CHECK(pid == cur_part);
    CHECK(index < size && index >= 0) << "Error: ScatterReudce, index out of range!";
    
    in_degs[index] += iptr[0];
    recv_list_nodes[i] = index;
    DType *optr = feat + index * feat_size;

    CHECK((i+1)*(feat_size+1) <= dim) << "Error: ScatterReudce, feat_size out of range!";
    CHECK((index+1)*feat_size <= fsize) << "Error: ScatterReudce, feat_size (fsize) out of range!";
    
    #pragma omp simd
    for (int32_t j=0; j<feat_size; j++)
      optr[j] += iptr[1 + j];
  }
  pos[0] += lim;
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelScatterReduceLR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray otf             = args[0];
  int64_t offsetf         = args[1];
  NDArray otn             = args[2];
  int64_t offsetn         = args[3];
  NDArray feat            = args[4];
  NDArray in_degs         = args[5];
  NDArray node_map        = args[6];
  int64_t dim             = args[7];
  int64_t feat_size       = args[8];
  int64_t num_parts       = args[9];
  NDArray recv_list_nodes = args[10];
  NDArray pos             = args[11];
  int64_t lim             = args[12];
  int64_t cur_part        = args[13];

  ATEN_FLOAT_TYPE_SWITCH(in_degs->dtype, IdType, "Id data", {
      ATEN_FLOAT_TYPE_SWITCH(otf->dtype, DType, "Feature data", {
          ScatterReduceLR<IdType, DType>(otf, offsetf, otn, offsetn,
                                         feat, in_degs, node_map,
                                         dim, feat_size, num_parts,
                                         recv_list_nodes, pos, lim,
                                         cur_part);
        });
    });
});


/*!
  \brief Root node gathers partial aggregates to be sent to leaf nodes.
  \param feat_a node features in current partition
  \param Nn number of nodes in the input graph
  \param send_feat_list_a buffer for gathered aggregates
  \param offset offset in the send_feat_list_a
  \param node IDs to be gathered
  \param lim number of nodes to be gathered
  \param in_degs_a node degree
  \param feat_size node feature vector size
  \param node_map_a node ID range for all the partitions
  \param num_parts number of mpi ranks or partitions
*/
template<typename DType>
void FdrpaGatherEmbRL(
  NDArray feat_a,
  int64_t Nn,
  NDArray send_feat_list_a,
  int64_t offset,
  NDArray recv_list_nodes_a,
  int64_t lim,
  NDArray in_degs_a,
  int32_t feat_size,
  NDArray node_map_a,
  int32_t num_parts) {
  
  DType*   feat            = feat_a.Ptr<DType>();
  DType*   send_feat_list  = send_feat_list_a.Ptr<DType>() + offset;
  int32_t* recv_list_nodes  = recv_list_nodes_a.Ptr<int32_t>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  DType*   in_degs         = in_degs_a.Ptr<DType>();
  
  int32_t pos = 0, pos_out = 0;

  #pragma omp parallel for
  for (int64_t i=0; i<lim; i++) {
    int64_t index = recv_list_nodes[i];
    CHECK(index < Nn) << "Error: FdrpaGartherRL, index of range of feat_shape!";
    DType *iptr = feat + index * feat_size;
    DType *optr = send_feat_list + i * (feat_size + 1);
    optr[0] = in_degs[index];
    
    #pragma omp simd
    for (int32_t k=0; k<feat_size; k++)
      optr[1 + k] = iptr[k];
  }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaGatherEmbRL")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray feat              = args[0];
  int64_t Nn                = args[1];
  NDArray send_feat_list    = args[2];
  int64_t offset            = args[3];
  NDArray recv_list_nodes   = args[4];
  int64_t lim               = args[5];
  NDArray in_degs           = args[6];
  int32_t feat_size         = args[7];
  NDArray node_map          = args[8];
  int32_t num_parts         = args[9];

  ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Feature data", {
      FdrpaGatherEmbRL<DType>(feat,
                              Nn,
                              send_feat_list,
                              offset,
                              recv_list_nodes,
                              lim,
                              in_degs,
                              feat_size,
                              node_map,
                              num_parts);
    });
});


/*!
  \brief Leaf nodes scatter reduce the recvd partial aggregates to local partial aggregates.
  \param otf_a recvd partial aggreagated from leafs
  \param offset offset in otf_a
  \param stn_a buffered leaf local nodes whose partial aggregate were communicated
  \param lim number of nodes recvd from a remote parition/mpi rank
  \param in_degs_a node degree
  \param feat_a node features in current partition
  \param node_map_a node ID range for all the partitions
  \param dim recvd partial aggregate size
  \param feat_size node feature vector size
  \param num_parts number of mpi ranks or partitions
*/
template<typename DType>
void ScatterReduceRL(
  NDArray otf_a,
  int64_t offset,
  NDArray stn_a,
  int64_t lim,
  NDArray in_degs_a,
  NDArray feat_a,
  NDArray node_map_a,
  int64_t dim,
  int64_t feat_size,
  int64_t num_parts) {
  
  DType*   otf             = otf_a.Ptr<DType>() + offset;
  int32_t* stn             = stn_a.Ptr<int32_t>();
  DType*   feat            = feat_a.Ptr<DType>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  DType*   in_degs         = in_degs_a.Ptr<DType>();

  int64_t fsize = feat_a.GetSize() >> 2;

  #pragma omp parallel for
  for (int64_t i=0; i<lim; i++) {
    DType   *iptr  = otf + i * (feat_size + 1);
    int64_t  index = stn[i];
    DType   *optr  = feat + index * feat_size;
    in_degs[index] = iptr[0];

    CHECK((i+1)*(1+feat_size) <= dim) << "Error: ScatterReduceRL, index out of range!";
    CHECK((index+1)*feat_size <= fsize) << "Error: ScatterReduceRL, index (fsize) out of range!";

    #pragma simd
    for (int32_t j=0; j<feat_size; j++)
      optr[j] = iptr[1+j];
  }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelScatterReduceRL")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray otf             = args[0];
  int64_t offset          = args[1];
  NDArray stn             = args[2];
  int64_t lim             = args[3];
  NDArray in_degs         = args[4];
  NDArray feat            = args[5];
  NDArray node_map        = args[6];
  int64_t dim             = args[7];
  int64_t feat_size       = args[8];
  int64_t num_parts       = args[9];

  ATEN_FLOAT_TYPE_SWITCH(otf->dtype, DType, "Feature data", {
      ScatterReduceRL<DType>(otf, offset, stn, lim,
                             in_degs, feat, node_map,
                             dim, feat_size, num_parts);
    });
});

template<typename IdType>
void FdrpaCommBuckets(
  NDArray adj_a,
  NDArray selected_nodes_a,
  NDArray ver2part_a,
  NDArray ver2part_index_a,
  NDArray node_map_a,
  NDArray buckets_a,
  NDArray lf_a,
  int32_t width,
  int32_t num_parts,
  int32_t cur_part) {
  
  IdType*  adj             = adj_a.Ptr<IdType>();
  int32_t* selected_nodes  = selected_nodes_a.Ptr<int32_t>();
  int32_t* ver2part       = ver2part_a.Ptr<int32_t>();
  int32_t* ver2part_index = ver2part_index_a.Ptr<int32_t>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  int32_t* buckets         = buckets_a.Ptr<int32_t>();
  int32_t* lf              = lf_a.Ptr<int32_t>();
  
  int32_t nthreads = omp_get_max_threads();
  int32_t *gbucket = (int32_t*) calloc (sizeof(int32_t), nthreads * num_parts);
  
  int32_t tnodes = 0;
  int32_t len = selected_nodes_a.GetSize() >> 2;
  
  #pragma omp parallel
  {
    int32_t tid = omp_get_thread_num();
    std::vector<std::pair<int32_t, int32_t> > cache(num_parts);
    #pragma omp for
    for(int64_t i=0; i<len; i++) {
      int64_t index = selected_nodes[i];
      int32_t lf_val = lf[index];
      CHECK(lf_val != -200);
      int32_t p = Ver2partition_<IdType>(lf_val, node_map, num_parts);
      
      IdType *ptr = adj + width * index;
      int32_t min_part = p;
      int32_t min_index = lf_val;
      
      if (min_part != cur_part) {
        // debug code
        bool flg = 1;
        for (int32_t j=0; j<width; j++) {
          if (ptr[j] >= 0) {
            if(ptr[j] == lf_val)
              flg = 0;
          }                
          else
            break;
        }
        CHECK(flg == 0);
        // debug code ends
        gbucket[tid*num_parts + min_part] ++;
        ver2part[i] = min_part;
        ver2part_index[i] = min_index;
      }
      else {
        ver2part[i] = -100;
      }
    }
  }
  // aggregate
  for (int32_t t=0; t<nthreads; t++) {
    for (int32_t p=0; p<num_parts; p++)
      buckets[p] += gbucket[t*num_parts + p];
  }
  for (int32_t p=0; p<num_parts; p++) tnodes += buckets[p];

  free(gbucket);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaCommBuckets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray adj            = args[0];
  NDArray selected_nodes = args[1];
  NDArray ver2part       = args[2];
  NDArray ver2part_index = args[3];
  NDArray node_map       = args[4];
  NDArray buckets        = args[5];
  NDArray lf             = args[6];
  int32_t width          = args[7];
  int32_t num_parts      = args[8];
  int32_t cur_part       = args[9];

  ATEN_ID_TYPE_SWITCH(adj->dtype, IdType, {
      FdrpaCommBuckets<IdType>(adj,
                               selected_nodes,
                               ver2part,
                               ver2part_index,
                               node_map,
                               buckets,
                               lf,
                               width,
                               num_parts,
                               cur_part);
    });
});


}  // namespace aten
}  // namespace dgl
