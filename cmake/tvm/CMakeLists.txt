set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -fPIC")
add_library(featgraph SHARED tvm_runtime.cc featgraph.cc)
target_include_directories(featgraph PRIVATE "../../third_party/tvm/include")
target_include_directories(featgraph PRIVATE "../../third_party/tvm/3rdparty/dmlc-core/include")
target_include_directories(featgraph PRIVATE "../../third_party/tvm/3rdparty/dlpack/include")
target_link_libraries(featgraph ${CUDA_CUDART_LIBRARY}
                                ${CUDA_CUDA_LIBRARY}
                                ${CUDA_NVRTC_LIBRARY})
install(TARGETS featgraph LIBRARY DESTINATION lib)