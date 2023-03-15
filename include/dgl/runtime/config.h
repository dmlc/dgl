/**
 *  Copyright (c) 2019 by Contributors
 * @file runtime/config.h
 * @brief DGL runtime config
 */

#ifndef DGL_RUNTIME_CONFIG_H_
#define DGL_RUNTIME_CONFIG_H_

namespace dgl {
namespace runtime {

class Config {
 public:
  static Config* Global() {
    static Config config;
    return &config;
  }

  // Enabling or disable use libxsmm for Spmm
  void EnableLibxsmm(bool);
  bool IsLibxsmmAvailable() const;

 private:
  Config();
  bool libxsmm_;
};

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_CONFIG_H_
