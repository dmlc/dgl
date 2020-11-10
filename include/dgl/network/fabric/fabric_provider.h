#pragma once

#include <dgl/network/common.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <dmlc/logging.h>
#include <netinet/in.h>
#include <rdma/fabric.h>

#include <string>

namespace dgl {
namespace network {

// struct sock_in ConvertToSockIn(const char *addr) {}

class FabricProvider {
 public:
  FabricProvider() {}

  FabricProvider(std::string prov_name) : prov_name(prov_name) {
    UniqueFabricPtr<struct fi_info> hints(fi_allocinfo());
    hints->ep_attr->type = FI_EP_RDM;  // Reliable Datagram
    hints->caps = FI_TAGGED | FI_MSG;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    // hints->domain_attr->caps = FI_LOCAL_COMM | FI_REMOTE_COMM;
    // strdup
    if (prov_name != "shm") {
      // hints->mode = FI_CONTEXT;
    }
    hints->domain_attr->av_type = FI_AV_TABLE;
    hints->fabric_attr->prov_name = strdup(prov_name.c_str());

    // fi_getinfo
    struct fi_info *info_;
    int ret =
      fi_getinfo(FABRIC_VERSION, nullptr, nullptr, 0, hints.get(), &info_);
    info.reset(info_);
    CHECK_NE(ret, -FI_ENODATA) << "Could not find any optimal provider";
    check_err(ret, "fi_getinfo failed");

    LOG(INFO) << "Found a fabric provider [" << 0 << "] "
              << info->fabric_attr->prov_name << ":"
              << fi_tostr(&info->addr_format, FI_TYPE_ADDR_FORMAT);

    LOG(INFO) << "Domain Name" << info->domain_attr->name;
    // struct fi_info *providers = info.get();
    // int i = 0;
    // while (providers) {
    // LOG(INFO) << "Found a fabric provider [" << i << "] "
    //           << providers->fabric_attr->prov_name << ":"
    //           << fi_tostr(&providers->addr_format, FI_TYPE_ADDR_FORMAT);
    //   i++;
    //   providers = providers->next;
    // }
  }

  static FabricProvider *CreateTcpProvider(const char *addr) {
    UniqueFabricPtr<struct fi_info> hints(fi_allocinfo());
    hints->ep_attr->type = FI_EP_RDM;  // Reliable Datagram
    hints->caps = FI_TAGGED | FI_MSG | FI_DIRECTED_RECV | FI_SOURCE;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->addr_format = FI_SOCKADDR_IN;
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    hints->domain_attr->av_type = FI_AV_TABLE;
    // hints->nic->link_attr->network_type = strdup("Ethernet");
    hints->fabric_attr->prov_name = strdup("udp");

    struct fi_info *info_;
    int ret = -1;
    if (addr) {
      std::vector<std::string> substring, hostport;
      SplitStringUsing(addr, "//", &substring);
      SplitStringUsing(substring[1], ":", &hostport);
      ret = fi_getinfo(FABRIC_VERSION, hostport[0].c_str(), hostport[1].c_str(),
                       FI_SOURCE, hints.get(), &info_);
    } else {
      ret =
        fi_getinfo(FABRIC_VERSION, nullptr, nullptr, 0, hints.get(), &info_);
    };

    // fi_getinfo
    FabricProvider *provider = new FabricProvider();
    provider->prov_name = "udp";
    provider->info.reset(info_);

    CHECK_NE(ret, -FI_ENODATA) << "Could not find any optimal provider";
    check_err(ret, "fi_getinfo failed");
    LOG(INFO) << "CTRL Domain Name" << provider->info->domain_attr->name;
    // LOG(INFO) << "CTRL NIC Name " << provider->info->nic->device_attr->name;
    // LOG(INFO) << "NIC1 Name" << info_->nic;
    // LOG(INFO) << "NIC2 Name" << info_->nic->device_attr;
    // LOG(INFO) << "NIC Name " << info_->nic->device_attr->name;
    // LOG(INFO) << "CTRL NIC Name " << provider->info->nic->device_attr->name;
    return provider;
  }
  std::string prov_name;

  UniqueFabricPtr<fi_info> info;
};
}  // namespace network
}  // namespace dgl