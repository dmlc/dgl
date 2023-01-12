/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/runtime/c_object_api.h
 *
 * @brief DGL Object C API, used to extend and prototype new CAPIs.
 *
 * @note Most API functions are registerd as PackedFunc and
 *  can be grabbed via DGLFuncGetGlobal
 */
#ifndef DGL_RUNTIME_C_OBJECT_API_H_
#define DGL_RUNTIME_C_OBJECT_API_H_

#include "./c_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief handle to object */
typedef void* ObjectHandle;

/**
 * @brief free the object handle
 * @param handle The object handle to be freed.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLObjectFree(ObjectHandle handle);

/**
 * @brief Convert type key to type index.
 * @param type_key The key of the type.
 * @param out_index the corresponding type index.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLObjectTypeKey2Index(const char* type_key, int* out_index);

/**
 * @brief Get runtime type index of the object.
 * @param handle the object handle.
 * @param out_index the corresponding type index.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLObjectGetTypeIndex(ObjectHandle handle, int* out_index);

/**
 * @brief get attributes given key
 * @param handle The object handle
 * @param key The attribute name
 * @param out_value The attribute value
 * @param out_type_code The type code of the attribute.
 * @param out_success Whether get is successful.
 * @return 0 when success, -1 when failure happens
 * @note API calls always exchanges with type bits=64, lanes=1
 */
DGL_DLL int DGLObjectGetAttr(
    ObjectHandle handle, const char* key, DGLValue* out_value,
    int* out_type_code, int* out_success);

/**
 * @brief get attributes names in the object.
 * @param handle The object handle
 * @param out_size The number of functions
 * @param out_array The array of function names.
 * @return 0 when success, -1 when failure happens
 */
DGL_DLL int DGLObjectListAttrNames(
    ObjectHandle handle, int* out_size, const char*** out_array);
#ifdef __cplusplus
}  // DGL_EXTERN_C
#endif
#endif  // DGL_RUNTIME_C_OBJECT_API_H_
