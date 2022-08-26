/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/config.h
 * \brief DGL aten config
 */


#ifndef DGL_ARRAY_CONFIG_H_
#define DGL_ARRAY_CONFIG_H_

namespace dgl {
namespace aten {

class Config {
 public:
  static Config* Global() {
    static Config config;
    return &config;
  }

  // Enabling or disable use libxsmm for Spmm
  void enableLibxsmm(bool);
  bool isLibxsmmAvailable() const;

 private:
  Config() =  default;
  bool _libxsmm = true;
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CONFIG_H_

