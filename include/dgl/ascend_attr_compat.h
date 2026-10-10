#ifndef DGL_ASCEND_ATTR_COMPAT_H_
#define DGL_ASCEND_ATTR_COMPAT_H_

// CANN 8.3.RC1 (25.5.x) ships an acl_rt.h that lacks the
// ACL_DEV_ATTR_CUBE_CORE_NUM / ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE
// enum values that were added in CANN 8.5.0.  The numeric values are
// stable device-info attribute IDs, verified at runtime on a 910B3:
//   ACL_DEV_ATTR_CUBE_CORE_NUM             = 102
//   ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE = 202
// If aclrtGetDeviceInfo does not recognise the ID it returns an error and
// the callers fall back to their built-in defaults, so these values are
// safe even if a future CANN renumbers them.
//
// NB: This header intentionally does NOT include <acl/acl_rt.h>.  It is
// force-included into every TU of the dgl target; translation units that
// also include torch_npu's bundled acl headers would otherwise hit
// duplicate enum definitions (e.g. ACL_MODEL_RI_CAPTURE_MODE_GLOBAL)
// between the two copies of the ACL headers.  aclrtDevAttr is always
// available by the time these macros are expanded in the .cc sources.

#ifndef ACL_DEV_ATTR_CUBE_CORE_NUM
#define ACL_DEV_ATTR_CUBE_CORE_NUM static_cast<aclrtDevAttr>(102)
#endif

#ifndef ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE
#define ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE static_cast<aclrtDevAttr>(202)
#endif

#endif  // DGL_ASCEND_ATTR_COMPAT_H_
