#include <torch/script.h>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "cnpy.h"

int main(int argc, char *argv[]) {
  // std::string f = "/home/ubuntu/dgl/dgl/datasets/arr7.npy";

  //"/home/ubuntu/dgl/dgl/datasets/ogbl-lsc-mag240m/preprocessed/features/paper-feat.npy"
  const std::string filename = argv[1];
  std::string f = filename;
  // cnpy::test_pread_seq_full(f, 1, 16);
  cnpy::NpyArray arr(f);
  arr.print_npy_header();
  arr.load_all();

  auto idx = torch::tensor({10, 1003, 1004, 10000});
  // auto aaaaa = torch::from_blob(idx, {3}, torch::kInt);

  // auto ret = arr.index_select(idx);

  // std::cout << ret;

  // auto ret = arr.index_select_pread(idx);
  // auto ret = arr.index_select_aio(idx);  // some bugs
  auto ret = arr.index_select_iouring(idx);
  // auto ret = arr.index_select_pread_single(idx);

  std::cout << ret << std::endl;
}