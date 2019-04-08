/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/runtime/ndarray.h
 * \brief shared memory management.
 */
#ifndef DGL_RUNTIME_SHARED_MEM_H_
#define DGL_RUNTIME_SHARED_MEM_H_

#include <string>

namespace dgl {
namespace runtime {

#ifndef _WIN32
/*
 * \brief This class owns shared memory.
 *
 * When the object is gone, the shared memory will also be destroyed.
 * When the shared memory is destroyed, the file corresponding to
 * the shared memory is removed.
 */
class SharedMemory {
  /*
   * \brief whether the shared memory is owned by the object.
   *
   * If shared memory is created in the object, it'll be owned by the object
   * and will be responsible for deleting it when the object is destroyed.
   */
  bool own;
  /*
   * \brief the name of the object.
   *
   * In Unix, shared memory is identified by a file. Thus, `name` is actually
   * the file name that identifies the shared memory.
   */
  std::string name;
  /* \brief the file descripter of the shared memory. */
  int fd;
  /* \brief the address of the shared memory. */
  void *ptr;
  /* \brief the size of the shared memory. */
  size_t size;

 public:
  /*
   * \brief constructor of the shared memory.
   * \param name The file corresponding to the shared memory.
   */
  explicit SharedMemory(const std::string &name);
  /*
   * \brief destructor of the shared memory.
   * It deallocates the shared memory and removes the corresponding file.
   */
  ~SharedMemory();
  /*
   * \brief create shared memory.
   * It creates the file and shared memory.
   * \param size the size of the shared memory.
   * \return the address of the shared memory
   */
  void *create_new(size_t size);
  /*
   * \brief allocate shared memory that has been created.
   * \param size the size of the shared memory.
   * \return the address of the shared memory
   */
  void *open(size_t size);
};
#endif  // _WIN32

}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_SHARED_MEM_H_
