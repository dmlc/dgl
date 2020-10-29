#pragma once

#include <dmlc/logging.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <sstream>
#include <string>

#define check_err(ret, msg)                           \
  do {                                                \
    if (ret != 0) {                                   \
      LOG(FATAL) << msg << ". Return Code: " << ret   \
                 << ". ERROR: " << fi_strerror(-ret); \
    }                                                 \
  } while (false)

namespace dgl {
namespace network {

static const int FABRIC_VERSION = FI_VERSION(1, 8);

struct FabricDeleter {
  void operator()(fi_info* info) { fi_freeinfo(info); }
  void operator()(fid* fid) { fi_close(fid); }
  void operator()(fid_domain* fid) { fi_close((fid_t)fid); }
  void operator()(fid_fabric* fid) { fi_close((fid_t)fid); }
  void operator()(fid_cq* fid) { fi_close((fid_t)fid); }
  void operator()(fid_av* fid) { fi_close((fid_t)fid); }
  void operator()(fid_ep* fid) { fi_close((fid_t)fid); }
  void operator()(fid_eq* fid) { fi_close((fid_t)fid); }
};

template <typename T>
using UniqueFabricPtr = std::unique_ptr<T, FabricDeleter>;

struct FabricAddr {
  // endpoint name
  char name[64] = {};
  // length of endpoint name
  size_t len = sizeof(name);

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < len; i++) {
      ss << std::to_string(name[i]) << ",";
    }
    ss << "]";
    return ss.str();
  }

  std::string str() const { return std::string(name, len); }

  void CopyFrom(const char* ep_name, const size_t ep_name_len) {
    len = ep_name_len;
    memcpy(name, ep_name, sizeof(name));
  }

  void CopyTo(char* ep_name, size_t* ep_name_len) {
    *(ep_name_len) = len;
    memcpy(ep_name, name, sizeof(name));
  }
};
}  // namespace network
}  // namespace dgl