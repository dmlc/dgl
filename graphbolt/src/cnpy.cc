#include "cnpy.h"

#include <aio.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <liburing.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <regex>
#include <stdexcept>

#define ALIGNMENT 4096

uint64_t time_diff(struct timespec *start, struct timespec *end) {
  return (end->tv_sec - start->tv_sec) * 1000000000 +
         (end->tv_nsec - start->tv_nsec);
}
void cnpy::test_pread_seq_full(
    const std::string &file_path, int thread_num, uint32_t block_size) {
  block_size = block_size * 1024;
  std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
  std::size_t file_size = is.tellg();
  is.close();

  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);

  int align_size =
      (file_size / block_size + (file_size % block_size != 0)) * block_size;

  char *buf = (char *)aligned_alloc(block_size, align_size);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  if (pread(fd, buf, align_size, 0) == -1) {
    printf("error\n");
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto duration = time_diff(&start, &end) / 1000000.0;
  uint64_t bandwidth = (align_size / (1024 * 1024)) / (duration / 1000);
  uint64_t *array = (uint64_t *)buf;
  uint64_t sum = 0;
  for (size_t i = 0; i < file_size / sizeof(uint64_t); ++i) {
    sum ^= array[i];
  }
  printf("%lu, %.3fms, %luMB/s\n", sum, duration, bandwidth);
  free(buf);
  close(fd);
}

void cnpy::NpyArray::load_all() {
  std::ifstream is(filename, std::ifstream::binary | std::ifstream::ate);
  std::size_t file_size = is.tellg();
  is.close();

  printf("file size: %lu\n", file_size);

  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    printf("open error\n");
  }

  int align_size =
      (file_size / ALIGNMENT + (file_size % ALIGNMENT != 0)) * ALIGNMENT;

  data_holder = (char *)aligned_alloc(ALIGNMENT, align_size);

  if (pread(fd, data_holder, align_size, 0) == -1) {
    printf("error\n");
  }
  close(fd);
}

torch::Tensor cnpy::NpyArray::index_select_pread_single(torch::Tensor idx) {
  int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;
  int64_t num_idx = idx.numel();

  char *read_buffer = (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  for (int64_t n = 0; n < num_idx; n++) {
    int64_t i = idx_data[n];
    int64_t offset = i * feature_size + prefix_len;
    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
    int64_t residual = offset - aligned_offset;
    int64_t read_size;

    if (residual + feature_size > ALIGNMENT) {
      read_size = ALIGNMENT * 2;
    } else {
      read_size = ALIGNMENT;
    }

    if (pread(feature_fd, read_buffer, read_size, aligned_offset) == -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
    memcpy(
        result_buffer + feature_size * n, read_buffer + residual, feature_size);
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options).clone();

  free(read_buffer);
  free(result_buffer);
  close(feature_fd);

  return result;
}

torch::Tensor cnpy::NpyArray::index_select_pread(torch::Tensor idx) {
  int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;
  int64_t num_idx = idx.numel();

  int numthd = 1;  // omp_get_num_procs();
  omp_set_num_threads(numthd);

  char *read_buffer = (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * numthd);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(numthd)
  for (int64_t n = 0; n < num_idx; n++) {
    int64_t i = idx_data[n];
    int64_t offset = i * feature_size + prefix_len;
    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
    int64_t residual = offset - aligned_offset;
    int64_t read_size;

    if (residual + feature_size > ALIGNMENT) {
      read_size = ALIGNMENT * 2;
    } else {
      read_size = ALIGNMENT;
    }

    if (pread(
            feature_fd, read_buffer + (ALIGNMENT * 2 * omp_get_thread_num()),
            read_size, aligned_offset) == -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
    memcpy(
        result_buffer + feature_size * n,
        read_buffer + (ALIGNMENT * 2 * omp_get_thread_num() + residual),
        feature_size);
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options).clone();

  free(read_buffer);
  free(result_buffer);
  close(feature_fd);

  return result;
}

torch::Tensor cnpy::NpyArray::index_select_aio(torch::Tensor idx) {
  int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;
  int64_t num_idx = idx.numel();

  int64_t group_size = 23;

  char *read_buffer =
      (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * group_size);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  aiocb *rd = (aiocb *)aligned_alloc(ALIGNMENT, sizeof(aiocb) * group_size);

  uint64_t residual[group_size];
  for (int64_t n = 0; n < num_idx; n += group_size) {
    for (int64_t m = 0; n + m < std::min(num_idx, n + group_size); m++) {
      int64_t i = idx_data[n + m];
      int64_t offset = i * feature_size + prefix_len;
      int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
      int64_t read_size;
      residual[m] = offset - aligned_offset;

      if (residual[m] + feature_size > ALIGNMENT) {
        read_size = ALIGNMENT * 2;
      } else {
        read_size = ALIGNMENT;
      }

      rd[m].aio_buf = read_buffer + (ALIGNMENT * 2 * m);
      rd[m].aio_fildes = feature_fd;
      rd[m].aio_nbytes = read_size;
      rd[m].aio_offset = aligned_offset;
      // printf("aio_buf %u %u\n", rd[m].aio_buf,
      //     uint64_t(rd[m].aio_buf) % ALIGNMENT);

      if (aio_read(&rd[m]) == -1) {
        perror("aio_read");
        exit(1);
      }
    }
    for (int64_t m = 0; n + m < std::min(num_idx, n + group_size); m++) {
      // float *tt = (float *)(read_buffer + (ALIGNMENT * 2 * m + residual[m]));
      // printf("read_buffer[0]: %u\n", tt[0]);
      while (aio_error(&rd[m]) == EINPROGRESS) {
      }
      memcpy(
          result_buffer + feature_size * (n + m),
          read_buffer + (ALIGNMENT * 2 * m + residual[m]), feature_size);
    }
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options).clone();

  free(read_buffer);
  free(result_buffer);
  close(feature_fd);
  free(rd);

  return result;
}

torch::Tensor cnpy::NpyArray::index_select_iouring(torch::Tensor idx) {
  int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;
  int64_t num_idx = idx.numel();

  int numthd = 8;
  omp_set_num_threads(numthd);

  int64_t group_size = 512;

  char *read_buffer =
      (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * 4 * group_size);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  struct io_uring ring;
  io_uring_queue_init(1024, &ring, 0);

  int64_t residual[group_size * 2];

  int64_t l[2] = {0, 0}, r[2] = {0, 0}, group = 0;
  r[group] = std::min(num_idx, l[group] + group_size);

  auto prepare_func = [&] {
  // printf("prepare group: %u\n", group);
#pragma omp parallel for num_threads(numthd)
    for (int64_t m = 0; m < r[group] - l[group]; m++) {
      int64_t i = idx_data[l[group] + m];
      int64_t offset = i * feature_size + prefix_len;
      int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
      int64_t read_size;
      residual[m + group * group_size] = offset - aligned_offset;

      if (residual[m + group * group_size] + feature_size > ALIGNMENT) {
        read_size = ALIGNMENT * 2;
      } else {
        read_size = ALIGNMENT;
      }

#pragma omp critical
      {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(
            sqe, feature_fd,
            read_buffer + (ALIGNMENT * 2 * m) +
                group * ALIGNMENT * 2 * group_size,
            read_size, aligned_offset);
      }
    }
    group ^= 1;
  };

  auto wait_func = [&] {
    // printf("wait group: %u\n", group);
    int64_t finish_num = 0;
    // printf("wait %lu %lu\n", r[group], l[group]);
    while (finish_num < r[group] - l[group]) {
      struct io_uring_cqe *cqe;
      if (io_uring_wait_cqe(&ring, &cqe) < 0) {
        perror("io_uring_wait_cqe");
        abort();
      }
      struct io_uring_cqe *cqes[group_size];
      int cqecount = io_uring_peek_batch_cqe(&ring, cqes, group_size);
      if (cqecount == -1) {
        perror("io_uring_peek_batch error\n");
        abort();
      }
      io_uring_cq_advance(&ring, cqecount);
      finish_num += cqecount;
      // printf("finsh num: %lu\n", finish_num);
    }
  };

  auto copy_func = [&] {
  // printf("copy group: %u\n", group);
#pragma omp parallel for num_threads(numthd)
    for (int64_t m = 0; m < r[group] - l[group]; m++) {
      memcpy(
          result_buffer + feature_size * (l[group] + m),
          read_buffer + group * ALIGNMENT * 2 * group_size +
              (ALIGNMENT * 2 * m + residual[m + group * group_size]),
          feature_size);
    }
  };
  // g=0 l=0 r=2
  prepare_func();  // pre g 0 g=1 l=2 r=4
  l[group] = l[group ^ 1] + group_size;
  r[group] = std::min(num_idx, l[group] + group_size);
  io_uring_submit(&ring);  // sub g 0
  // printf("submit group: %u\n", group ^ 1);

  for (; l[group] < num_idx;) {  //
    prepare_func();              // pre g 1  g=0
    wait_func();                 // wait g 0
    io_uring_submit(&ring);      // sub g 1
    // printf("submit group: %u\n", group ^ 1);
    copy_func();  // copy g 0
    l[group] = l[group ^ 1] + group_size;
    r[group] = std::min(num_idx, l[group] + group_size);  // l=4 r=4
  }
  group ^= 1;
  wait_func();
  copy_func();

  io_uring_queue_exit(&ring);

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options).clone();

  free(read_buffer);
  free(result_buffer);
  close(feature_fd);

  return result;
}

torch::Tensor cnpy::NpyArray::index_select_all(torch::Tensor idx) {
  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;

  int64_t num_idx = idx.numel();

  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  int64_t upper = 0;
  int64_t in_upper = 0;
  int64_t in_lower = 0;
  std::vector<int64_t> in_offset(num_idx);
  std::vector<int64_t> page_offset{};
  for (int64_t i = 0; i < num_idx; i++) {
    // from true offset to in-offset
    // printf("idx: %u\n", idx_data[i]);
    int64_t offset = idx_data[i] * feature_size + prefix_len;
    // printf("offset: %u\n", offset);
    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
    // printf("algn: %u\n", aligned_offset);
    int64_t residual = offset - aligned_offset;

    if (offset > upper) {
      in_offset[i] = in_upper + residual;
      upper = aligned_offset + ALIGNMENT;
      page_offset.push_back(aligned_offset);
      // printf("put page: %u\n", aligned_offset);
      in_lower = in_upper;
      in_upper += ALIGNMENT;
    } else {
      in_offset[i] = in_lower + residual;
    }
    if (residual + feature_size > ALIGNMENT) {
      upper += ALIGNMENT;
      in_lower = in_upper;
      in_upper += ALIGNMENT;
      page_offset.push_back(aligned_offset + ALIGNMENT);
      // printf("put page: %u\n", aligned_offset + ALIGNMENT);
    }
  }

  // printf("page offset: ");
  // for (auto i : page_offset) {
  //   printf("%u ", i);
  // }
  // printf("\n");
  // printf("in offset: ");
  // for (auto i : in_offset) {
  //   printf("%u ", i);
  // }
  // printf("\n");

  char *read_buffer =
      (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * page_offset.size());

  for (size_t i = 0; i < page_offset.size(); i++) {
    if (pread(fd, read_buffer + i * ALIGNMENT, ALIGNMENT, page_offset[i]) ==
        -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
  }

  // printf("page view\n");
  // for (uint i = 0; i < 10; i++) {
  //   printf("preview %.3f\n", reinterpret_cast<float *>(read_buffer + 80)[i]);
  // }

  for (uint64_t i = 0; i < in_offset.size(); i++) {
    memcpy(
        result_buffer + feature_size * i, read_buffer + in_offset[i],
        feature_size);
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

  free(read_buffer);
  close(fd);

  return result;
}

void cnpy::NpyArray::print_npy_header() {
  printf("prefix size: %lu\n", prefix_len);
  std::cout << "feature_shape:" << feature_shape << std::endl;
  printf(
      "file: %s\nword_size: %lu \nfeature_dim: %lu\nfortran_order: %u\n",
      filename.c_str(), word_size, feature_dim, fortran_order);
  for (size_t i = 0; i < shape.size(); i++) {
    printf("shape %lu\n", shape[i]);
  }
}

void cnpy::NpyArray::parse_npy_header(FILE *fp) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
  std::string header = fgets(buffer, 256, fp);
  assert(header[header.size() - 1] == '\n');
  prefix_len = 11 + header.size();

  size_t loc1, loc2;

  // fortran order
  loc1 = header.find("fortran_order");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

  // shape
  loc1 = header.find("(");
  loc2 = header.find(")");
  if (loc1 == std::string::npos || loc2 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: '(' or ')'");

  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  shape.clear();

  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  while (std::regex_search(str_shape, sm, num_regex)) {
    shape.emplace_back(std::stoi(sm[0].str()));
    str_shape = sm.suffix().str();
  }
  num_vals = feature_dim = 1;

  for (size_t i = 0; i < shape.size(); i++) {
    num_vals *= shape[i];
  }

  std::vector<int64_t> shape_vec;
  for (size_t i = 1; i < shape.size(); i++) {
    feature_dim *= shape[i];
    shape_vec.push_back(shape[i]);
  }

  at::TensorOptions opts = at::TensorOptions().dtype(at::kInt);
  feature_shape =
      torch::from_blob(shape_vec.data(), {int64_t(shape_vec.size())}, opts)
          .clone();

  loc1 = header.find("descr");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'descr'");
  loc1 += 9;
  bool littleEndian =
      (header[loc1] == '<' || header[loc1] == '|' ? true : false);
  assert(littleEndian);

  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  word_size = atoi(str_ws.substr(0, loc2).c_str());
}