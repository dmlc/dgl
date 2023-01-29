/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/serialize/heterograph_serialize.cc
 * @brief DGLHeteroGraph serialization implementation
 *
 * The storage structure is
 * {
 *   // MetaData Section
 *   uint64_t kDGLSerializeMagic
 *   uint64_t kVersion = 2
 *   uint64_t GraphType = kDGLHeteroGraph
 *   dgl_id_t num_graphs
 *   ** Reserved Area till 4kB **
 *
 *   uint64_t gdata_start_pos (This stores the start position of graph_data,
 * which is used to skip label dict part if unnecessary)
 *   vector<pair<string, NDArray>> label_dict (To store the dict[str, NDArray])
 *
 *   vector<HeteroGraphData> graph_datas;
 *   vector<dgl_id_t> graph_indices (start address of each graph)
 *   uint64_t size_of_graph_indices_vector (Used to seek to graph_indices
 * vector)
 *
 * }
 *
 * Storage of HeteroGraphData is
 * {
 *   HeteroGraphPtr ptr;
 *   vector<vector<pair<string, NDArray>>> node_tensors;
 *   vector<vector<pair<string, NDArray>>> edge_tensors;
 *   vector<string> ntype_name;
 *   vector<string> etype_name;
 * }
 *
 */
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../heterograph.h"
#include "./dglstream.h"
#include "./graph_serialize.h"
#include "dmlc/memory_io.h"

namespace dgl {
namespace serialize {

using namespace dgl::runtime;
using dmlc::SeekStream;
using dmlc::Stream;
using dmlc::io::FileSystem;
using dmlc::io::URI;

bool SaveHeteroGraphs(
    std::string filename, List<HeteroGraphData> hdata,
    const std::vector<NamedTensor> &nd_list, dgl_format_code_t formats) {
  auto fs = std::unique_ptr<DGLStream>(
      DGLStream::Create(filename.c_str(), "w", false, formats));
  CHECK(fs->IsValid()) << "File name " << filename << " is not a valid name";

  // Write DGL MetaData
  const uint64_t kVersion = 2;
  std::array<char, 4096> meta_buffer;

  // Write metadata into char buffer with size 4096
  dmlc::MemoryFixedSizeStream meta_fs_(meta_buffer.data(), 4096);
  auto meta_fs = static_cast<Stream *>(&meta_fs_);
  meta_fs->Write(kDGLSerializeMagic);
  meta_fs->Write(kVersion);
  meta_fs->Write(GraphType::kHeteroGraph);
  uint64_t num_graph = hdata.size();
  meta_fs->Write(num_graph);

  // Write metadata into files
  fs->Write(meta_buffer.data(), 4096);

  // Calculate label dict binary size
  std::string labels_blob;
  dmlc::MemoryStringStream label_fs_(&labels_blob);
  auto label_fs = static_cast<Stream *>(&label_fs_);
  label_fs->Write(nd_list);

  uint64_t gdata_start_pos =
      fs->Count() + sizeof(uint64_t) + labels_blob.size();

  // Write start position of gdata, which can be skipped when only reading gdata
  // And label dict
  fs->Write(gdata_start_pos);
  fs->Write(labels_blob.c_str(), labels_blob.size());

  std::vector<uint64_t> graph_indices(num_graph);

  // Write HeteroGraphData
  for (uint64_t i = 0; i < num_graph; ++i) {
    graph_indices[i] = fs->Count();
    auto gdata = hdata[i].sptr();
    fs->Write(gdata);
  }

  // Write indptr into string to count size
  std::string indptr_blob;
  dmlc::MemoryStringStream indptr_fs_(&indptr_blob);
  auto indptr_fs = static_cast<Stream *>(&indptr_fs_);
  indptr_fs->Write(graph_indices);

  uint64_t indptr_buffer_size = indptr_blob.size();
  fs->Write(indptr_blob);
  fs->Write(indptr_buffer_size);

  return true;
}

std::vector<HeteroGraphData> LoadHeteroGraphs(
    const std::string &filename, std::vector<dgl_id_t> idx_list) {
  auto fs = std::unique_ptr<SeekStream>(
      SeekStream::CreateForRead(filename.c_str(), false));
  CHECK(fs) << "File name " << filename << " is not a valid name";
  // Read DGL MetaData
  uint64_t magicNum, graphType, version, num_graph;
  fs->Read(&magicNum);
  fs->Read(&version);
  fs->Read(&graphType);
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  fs->Seek(4096);

  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  CHECK_EQ(version, 2) << "Invalid GraphType";
  CHECK_EQ(graphType, GraphType::kHeteroGraph) << "Invalid GraphType";

  uint64_t gdata_start_pos;
  fs->Read(&gdata_start_pos);
  // Skip labels part
  fs->Seek(gdata_start_pos);

  std::vector<HeteroGraphData> gdata_refs;
  if (idx_list.empty()) {
    // Read All Graphs
    gdata_refs.reserve(num_graph);
    for (uint64_t i = 0; i < num_graph; ++i) {
      HeteroGraphData gdata = HeteroGraphData::Create();
      auto hetero_data = gdata.sptr();
      fs->Read(&hetero_data);
      gdata_refs.push_back(gdata);
    }
  } else {
    uint64_t gdata_start_pos = fs->Tell();
    // Read Selected Graphss
    gdata_refs.reserve(idx_list.size());
    URI uri(filename.c_str());
    uint64_t filesize = FileSystem::GetInstance(uri)->GetPathInfo(uri).size;
    fs->Seek(filesize - sizeof(uint64_t));
    uint64_t indptr_buffer_size;
    fs->Read(&indptr_buffer_size);

    std::vector<uint64_t> graph_indices(num_graph);
    fs->Seek(filesize - sizeof(uint64_t) - indptr_buffer_size);
    fs->Read(&graph_indices);

    fs->Seek(gdata_start_pos);
    // Would be better if idx_list is sorted. However the returned the graphs
    // should be the same order as the idx_list
    for (uint64_t i = 0; i < idx_list.size(); ++i) {
      auto gid = idx_list[i];
      CHECK((gid < graph_indices.size()) && (gid >= 0))
          << "ID " << gid
          << " in idx_list is out of bound. Please check your idx_list.";
      fs->Seek(graph_indices[gid]);
      HeteroGraphData gdata = HeteroGraphData::Create();
      auto hetero_data = gdata.sptr();
      fs->Read(&hetero_data);
      gdata_refs.push_back(gdata);
    }
  }

  return gdata_refs;
}

std::vector<NamedTensor> LoadLabels_V2(const std::string &filename) {
  auto fs = std::unique_ptr<SeekStream>(
      SeekStream::CreateForRead(filename.c_str(), false));
  CHECK(fs) << "File name " << filename << " is not a valid name";
  // Read DGL MetaData
  uint64_t magicNum, graphType, version, num_graph;
  fs->Read(&magicNum);
  fs->Read(&version);
  fs->Read(&graphType);
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  fs->Seek(4096);

  uint64_t gdata_start_pos;
  fs->Read(&gdata_start_pos);

  std::vector<NamedTensor> labels_list;
  fs->Read(&labels_list);

  return labels_list;
}

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_MakeHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef hg = args[0];
      List<Map<std::string, Value>> ndata = args[1];
      List<Map<std::string, Value>> edata = args[2];
      List<Value> ntype_names = args[3];
      List<Value> etype_names = args[4];
      *rv = HeteroGraphData::Create(
          hg.sptr(), ndata, edata, ntype_names, etype_names);
    });

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_SaveHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      List<HeteroGraphData> hgdata = args[1];
      Map<std::string, Value> nd_map = args[2];
      List<Value> formats = args[3];
      std::vector<SparseFormat> formats_vec;
      for (const auto &val : formats) {
        formats_vec.push_back(ParseSparseFormat(val->data));
      }
      const auto formats_code = SparseFormatsToCode(formats_vec);
      std::vector<NamedTensor> nd_list;
      for (auto kv : nd_map) {
        NDArray ndarray = static_cast<NDArray>(kv.second->data);
        nd_list.emplace_back(kv.first, ndarray);
      }
      *rv = dgl::serialize::SaveHeteroGraphs(
          filename, hgdata, nd_list, formats_code);
    });

DGL_REGISTER_GLOBAL(
    "data.heterograph_serialize._CAPI_GetGindexFromHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphData hdata = args[0];
      *rv = HeteroGraphRef(hdata->gptr);
    });

DGL_REGISTER_GLOBAL(
    "data.heterograph_serialize._CAPI_GetEtypesFromHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphData hdata = args[0];
      List<Value> etype_names;
      for (const auto &name : hdata->etype_names) {
        etype_names.push_back(Value(MakeValue(name)));
      }
      *rv = etype_names;
    });

DGL_REGISTER_GLOBAL(
    "data.heterograph_serialize._CAPI_GetNtypesFromHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphData hdata = args[0];
      List<Value> ntype_names;
      for (auto name : hdata->ntype_names) {
        ntype_names.push_back(Value(MakeValue(name)));
      }
      *rv = ntype_names;
    });

DGL_REGISTER_GLOBAL(
    "data.heterograph_serialize._CAPI_GetNDataFromHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphData hdata = args[0];
      List<List<Value>> ntensors;
      for (auto tensor_list : hdata->node_tensors) {
        List<Value> nlist;
        for (const auto &kv : tensor_list) {
          nlist.push_back(Value(MakeValue(kv.first)));
          nlist.push_back(Value(MakeValue(kv.second)));
        }
        ntensors.push_back(nlist);
      }
      *rv = ntensors;
    });

DGL_REGISTER_GLOBAL(
    "data.heterograph_serialize._CAPI_GetEDataFromHeteroGraphData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphData hdata = args[0];
      List<List<Value>> etensors;
      for (auto tensor_list : hdata->edge_tensors) {
        List<Value> elist;
        for (const auto &kv : tensor_list) {
          elist.push_back(Value(MakeValue(kv.first)));
          elist.push_back(Value(MakeValue(kv.second)));
        }
        etensors.push_back(elist);
      }
      *rv = etensors;
    });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadLabels_V2")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string filename = args[0];
      auto labels_list = LoadLabels_V2(filename);
      Map<std::string, Value> rvmap;
      for (auto kv : labels_list) {
        rvmap.Set(kv.first, Value(MakeValue(kv.second)));
      }
      *rv = rvmap;
    });

}  // namespace serialize
}  // namespace dgl
