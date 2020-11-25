set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -fPIC")
add_library(tvm_runtime SHARED tvm_runtime.cc featgraph.cc)
target_include_directories(tvm_runtime PRIVATE "../../third_party/tvm/include")
target_include_directories(tvm_runtime PRIVATE "../../third_party/tvm/3rdparty/dmlc-core/include")
target_include_directories(tvm_runtime PRIVATE "../../third_party/tvm/3rdparty/dlpack/include")
target_link_libraries(tvm_runtime ${CUDA_CUDART_LIBRARY}
                                  ${CUDA_CUDA_LIBRARY}
                                  ${CUDA_NVRTC_LIBRARY})
install(TARGETS tvm_runtime LIBRARY DESTINATION lib)