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

  auto ret = arr.index_select(idx);

  std::cout << ret;

  // std::complex<double> *loaded_data = arr.data<std::complex<double>>();
  // int i, j;
  // i = 0, j = 0;
  // std::cout << loaded_data[i * Ny + j] << std::endl;
  // j = 1;
  // std::cout << loaded_data[i * Ny + j] << std::endl;
  // i = 1;
  // std::cout << loaded_data[i * Ny + j] << std::endl;
}