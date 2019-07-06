/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include "./array_op.h"
#include "./arith.h"
#include "./common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {

IdArray NewIdArray(int64_t length, DLContext ctx, uint8_t nbits) {
  return IdArray::Empty({length}, DLDataType{kDLInt, nbits, 1}, ctx);
}

BoolArray NewBoolArray(int64_t length, DLContext ctx) {
  return BoolArray::Empty({length}, DLDataType{kDLInt, 64, 1}, ctx);
}

IdArray VecToIdArray(const std::vector<int32_t>& vec, DLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DLContext{kDLCPU, 0}, 32);
  std::copy(vec.begin(), vec.end(), static_cast<int32_t*>(ret->data));
  return ret.CopyTo(ctx);
}

IdArray VecToIdArray(const std::vector<int64_t>& vec, DLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DLContext{kDLCPU, 0}, 64);
  std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(ret->data));
  return ret.CopyTo(ctx);
}

IdArray Clone(IdArray arr) {
  IdArray ret = NewIdArray(arr->shape[0]);
  ret.CopyFrom(arr);
  return ret;
}

IdArray AsNumBits(IdArray arr, uint8_t bits) {
  IdArray ret;
  ATEN_XPU_SWITCH(arr->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(arr->dtype, IdType, {
      ret = impl::AsNumBits<XPU, IdType>(arr, bits);
    });
  });
  return ret;
}

IdArray Add(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_EQ(lhs->ctx, rhs->ctx) << "Both operands should have the same device context";
  CHECK_EQ(lhs->dtype, rhs->dtype) << "Both operands should have the same dtype";
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Add>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Sub(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_EQ(lhs->ctx, rhs->ctx) << "Both operands should have the same device context";
  CHECK_EQ(lhs->dtype, rhs->dtype) << "Both operands should have the same dtype";
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Sub>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Mul(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_EQ(lhs->ctx, rhs->ctx) << "Both operands should have the same device context";
  CHECK_EQ(lhs->dtype, rhs->dtype) << "Both operands should have the same dtype";
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Mul>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Div(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_EQ(lhs->ctx, rhs->ctx) << "Both operands should have the same device context";
  CHECK_EQ(lhs->dtype, rhs->dtype) << "Both operands should have the same dtype";
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Div>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Add(IdArray lhs, dgl_id_t rhs) {
  IdArray ret;
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Add>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Sub(IdArray lhs, dgl_id_t rhs) {
  IdArray ret;
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Sub>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Mul(IdArray lhs, dgl_id_t rhs) {
  IdArray ret;
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Mul>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Div(IdArray lhs, dgl_id_t rhs) {
  IdArray ret;
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Div>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Add(dgl_id_t lhs, IdArray rhs) {
  return Add(rhs, lhs);
}

IdArray Sub(dgl_id_t lhs, IdArray rhs) {
  IdArray ret;
  ATEN_XPU_SWITCH(rhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(rhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Sub>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Mul(dgl_id_t lhs, IdArray rhs) {
  return Mul(rhs, lhs);
}

IdArray Div(dgl_id_t lhs, IdArray rhs) {
  IdArray ret;
  ATEN_XPU_SWITCH(rhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(rhs->dtype, IdType, {
      ret = impl::BinaryElewise<XPU, IdType, arith::Div>(lhs, rhs);
    });
  });
  return ret;
}

IdArray HStack(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_EQ(lhs->ctx, rhs->ctx) << "Both operands should have the same device context";
  CHECK_EQ(lhs->dtype, rhs->dtype) << "Both operands should have the same dtype";
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::HStack<XPU, IdType>(lhs, rhs);
    });
  });
  return ret;
}

IdArray Full(int32_t val, int64_t length, DLContext ctx) {
  IdArray ret;
  ATEN_XPU_SWITCH(ctx.device_type, XPU, {
    ret = impl::Full<XPU, int32_t>(val, length, ctx);
  });
  return ret;
}

IdArray Full(int64_t val, int64_t length, DLContext ctx) {
  IdArray ret;
  ATEN_XPU_SWITCH(ctx.device_type, XPU, {
    ret = impl::Full<XPU, int64_t>(val, length, ctx);
  });
  return ret;
}

IdArray Concat(const std::vector<IdArray>& arrays) {
  // TODO
}


///////////////////////// CSR routines //////////////////////////

bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  bool ret = false;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRIsNonZero<XPU, IdType, DType>(csr, row, col);
  });
  return ret;
}

int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  int64_t ret = 0;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRGetRowNNZ<XPU, IdType, DType>(csr, row);
  });
  return ret;
}

NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  NDArray ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRGetRowColumnIndices<XPU, IdType, DType>(csr, row);
  });
  return ret;
}

NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  NDArray ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRGetRowData<XPU, IdType, DType>(csr, row);
  });
  return ret;
}

NDArray CSRGetData(CSRMatrix csr, int64_t row, int64_t col) {
  NDArray ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRGetData<XPU, IdType, DType>(csr, row, col);
  });
  return ret;
}

NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols) {
  NDArray ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRGetData<XPU, IdType, DType>(csr, rows, cols);
  });
  return ret;
}

std::vector<NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, NDArray rows, NDArray cols) {
  std::vector<NDArray> ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRGetDataAndIndices<XPU, IdType, DType>(csr, rows, cols);
  });
  return ret;
}

CSRMatrix CSRTranspose(CSRMatrix csr) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRTranspose<XPU, IdType, DType>(csr);
  });
  return ret;
}

COOMatrix CSRToCOO(CSRMatrix csr, bool data_as_order) {
  COOMatrix ret;
  if (data_as_order) {
    ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, {
      ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
        ret = impl::CSRToCOODataAsOrder<XPU, IdType>(csr);
      });
    });
  } else {
    ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, {
      ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
        ret = impl::CSRToCOO<XPU, IdType>(csr);
      });
    });
  }
  return ret;
}

CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRSliceRows<XPU, IdType, DType>(csr, start, end);
  });
  return ret;
}

CSRMatrix CSRSliceMatrix(CSRMatrix csr, NDArray rows, NDArray cols) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, DType, {
    ret = impl::CSRSliceMatrix<XPU, IdType, DType>(csr, rows, cols);
  });
  return ret;
}

}  // namespace aten
}  // namespace dgl
