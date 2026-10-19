// ============================================================================
// CANN 版本兼容头（兼容8.3.RC1, CANN 升级为 8.5 后可回退）
//
// 背景：主仓 ascend host 文件（coo2csr.cc / csr_row_wise_sampling_biased.cc /
// csr_row_wise_sampling_uniform.cc / csr_row_wise_sampling_uniform_fused.cc /
// csr_row_wise_topk.cc / spmm.cc）按 CANN 8.5 的设备属性枚举名编写
// （ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE / ACL_DEV_ATTR_CUBE_CORE_NUM，
// 见 commit 908fa43f「修复ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE宏在cann8.5版本不支持问题」），
// 而当前构建环境 CANN 8.3.RC1 的 acl_rt.h 仅有
// ACL_DEV_ATTR_AICPU_CORE_NUM(1) / ACL_DEV_ATTR_AICORE_CORE_NUM(101) /
// ACL_DEV_ATTR_VECTOR_CORE_NUM(201)。
//
// 修复语义：8.3 驱动不支持 8.5 的属性查询，aclrtGetDeviceInfo 返回错误 →
// 调用方按既有回退逻辑取默认值（QueryUbAvailableBytes 回退 kDefaultUbBytes、
// GetCubeCoreCount 回退 24），与「查询失败回退」路径完全一致，不改任何计算逻辑。
// 数值沿用 8.5 枚举序推测值（AICORE=101→CUBE=102、VECTOR=201→LOCAL_MEM=202）；
// 8.3 不识别该属性编号，运行期不会返回该编号的有效数据。8.5 环境由 acl_rt.h
// 原生定义同名枚举，本头 #ifndef 守卫保证不覆盖（宏值等于枚举值，语义一致）。
//
// 用法：在 `#include <acl/acl_rt.h>` 之后 include 本头。
// ============================================================================
#ifndef DGL_ARRAY_ASCEND_ACL_ATTR_COMPAT_H_
#define DGL_ARRAY_ASCEND_ACL_ATTR_COMPAT_H_

// CANN 8.5 引入；8.3.RC1 缺失（仅编译期占位，运行期查询失败走默认值）
// C 风格 cast 保持 aclrtGetDeviceInfo 的 aclrtDevAttr 形参类型匹配
#ifndef ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE
#define ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE ((aclrtDevAttr)202)
#endif

#ifndef ACL_DEV_ATTR_CUBE_CORE_NUM
#define ACL_DEV_ATTR_CUBE_CORE_NUM ((aclrtDevAttr)102)
#endif

#endif  // DGL_ARRAY_ASCEND_ACL_ATTR_COMPAT_H_
