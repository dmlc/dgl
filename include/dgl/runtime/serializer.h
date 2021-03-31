/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/runtime/serializer.h
 * \brief Serializer extension to support DGL data types
 *  Include this file to enable serialization of DLDataType, DLContext
 */
#ifndef DGL_RUNTIME_SERIALIZER_H_
#define DGL_RUNTIME_SERIALIZER_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include "c_runtime_api.h"
#include "smart_ptr_serializer.h"

namespace dmlc {
namespace serializer {

template <>
struct Handler<DLDataType> {
  inline static void Write(Stream *strm, const DLDataType &dtype) {
    Handler<uint8_t>::Write(strm, dtype.code);
    Handler<uint8_t>::Write(strm, dtype.bits);
    Handler<uint16_t>::Write(strm, dtype.lanes);
  }
  inline static bool Read(Stream *strm, DLDataType *dtype) {
    if (!Handler<uint8_t>::Read(strm, &(dtype->code))) return false;
    if (!Handler<uint8_t>::Read(strm, &(dtype->bits))) return false;
    if (!Handler<uint16_t>::Read(strm, &(dtype->lanes))) return false;
    return true;
  }
};

template <>
struct Handler<DLContext> {
  inline static void Write(Stream *strm, const DLContext &ctx) {
    int32_t device_type = static_cast<int32_t>(ctx.device_type);
    Handler<int32_t>::Write(strm, device_type);
    Handler<int32_t>::Write(strm, ctx.device_id);
  }
  inline static bool Read(Stream *strm, DLContext *ctx) {
    int32_t device_type = 0;
    if (!Handler<int32_t>::Read(strm, &(device_type))) return false;
    ctx->device_type = static_cast<DLDeviceType>(device_type);
    if (!Handler<int32_t>::Read(strm, &(ctx->device_id))) return false;
    return true;
  }
};

}  // namespace serializer
}  // namespace dmlc
#endif  // DGL_RUNTIME_SERIALIZER_H_
