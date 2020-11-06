#pragma once

#include <dgl/network/fabric/fabric_provider.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <rdma/fabric.h>
#include <rdma/fi_eq.h>

#include <string>

namespace dgl {
namespace network {

class FabricContext {
 public:
  // fabric top-level object
  UniqueFabricPtr<struct fid_fabric> fabric;
  // domains which maps to a specific local network interface adapter
  UniqueFabricPtr<struct fid_domain> domain;
  // completion queue
  UniqueFabricPtr<struct fid_cq> txcq;
  UniqueFabricPtr<struct fid_cq> rxcq;
  // address vector
  UniqueFabricPtr<struct fid_av> av;
  // the endpoint
  UniqueFabricPtr<struct fid_ep> ep;
  // the event queue
  UniqueFabricPtr<struct fid_eq> eq;
  // endpoint name
  struct FabricAddr addr;
  // readable endpoint name
  struct FabricAddr readable_addr;
  // maximum tag
  uint64_t max_tag;

  std::shared_ptr<FabricProvider> fabric_provider;

 public:
  explicit FabricContext(std::shared_ptr<FabricProvider> fprovider)
      : fabric_provider(fprovider) {
    struct fi_info *info = fprovider->info.get();
    struct fi_cq_attr cq_attr = {};
    struct fi_av_attr av_attr = {};
    struct fi_eq_attr eq_attr = {};

    // fi_fabric: create fabric
    struct fid_fabric *fabric_;
    int ret = fi_fabric(info->fabric_attr, &fabric_, nullptr);
    check_err(ret, "Couldn't open a fabric provider");
    fabric.reset(fabric_);

    // fi_domain: create domain
    struct fid_domain *domain_;
    ret = fi_domain(fabric.get(), info, &domain_, nullptr);
    check_err(ret, "Couldn't open a fabric access domain");
    domain.reset(domain_);

    // fi_av_open: create address vector
    av_attr.type = FI_AV_TABLE;
    struct fid_av *av_;
    ret = fi_av_open(domain.get(), &av_attr, &av_, nullptr);
    av.reset(av_);
    check_err(ret, "Couldn't open AV");

    // fi_eq_open: open event queue
    struct fid_eq *eq_;
    ret = fi_eq_open(fabric.get(), &eq_attr, &eq_, nullptr);
    check_err(ret, "Couldn't open an event queue");
    eq.reset(eq_);

    // fi_cq_open: open completion queue
    struct fid_cq *rxcq_, *txcq_;
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    ret = fi_cq_open(domain.get(), &cq_attr, &rxcq_, nullptr);
    rxcq.reset(rxcq_);
    check_err(ret, "Couldn't open CQ");

    ret = fi_cq_open(domain.get(), &cq_attr, &txcq_, nullptr);
    txcq.reset(txcq_);
    check_err(ret, "Couldn't open CQ");

    // fi_endpoint: create transport level communication endpoint(s)
    struct fid_ep *ep_;
    ret = fi_endpoint(domain.get(), info, &ep_, nullptr);
    ep.reset(ep_);
    check_err(ret, "Couldn't allocate endpoint");

    // fi_ep_bind: bind CQ and AV to the endpoint
    ret = fi_ep_bind(ep.get(), (fid_t)rxcq.get(), FI_RECV);
    check_err(ret, "Couldn't bind EP-RXCQ");
    ret = fi_ep_bind(ep.get(), (fid_t)txcq.get(), FI_SEND);
    check_err(ret, "Couldn't bind EP-TXCQ");
    ret = fi_ep_bind(ep.get(), (fid_t)av.get(), 0);
    check_err(ret, "Couldn't bind EP-AV");
    ret = fi_ep_bind(ep.get(), (fid_t)eq.get(), 0);
    check_err(ret, "Couldn't bind EP-EQ");

    // fi_enable: enable endpoint for communication
    ret = fi_enable(ep.get());
    check_err(ret, "Couldn't enable endpoint");

    // fi_getname: get endpoint name
    ret = fi_getname((fid_t)ep.get(), addr.name, &addr.len);
    check_err(ret, "Call to fi_getname() failed");
    LOG(INFO) << "Endpoint created raw: " << std::string(addr.name, addr.len)
              << " readable endpoint = "
              << std::string(readable_addr.name, readable_addr.len);

    // fi_av_straddr: human readable name
    fi_av_straddr(av.get(), addr.name, readable_addr.name, &readable_addr.len);
    LOG(INFO) << "Endpoint created: " << addr.DebugStr()
              << " readable endpoint = "
              << std::string(readable_addr.name, readable_addr.len);
  }
};

}  // namespace network
}  // namespace dgl