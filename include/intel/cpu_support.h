/*!
 *  Copyright (c) 2019 by Contributors
 * \file intel/cpu_support.h
 * \brief Intel CPU support
 * \author Pawel Piotrowicz <pawel.piotrowicz@intel.com>
 */
#ifndef INTEL_CPU_SUPPORT_H_
#define INTEL_CPU_SUPPORT_H_
#include <memory>
#include <tuple>
#include <type_traits>

#include "dmlc/logging.h"
#include "meta_utils.h"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace dgl {

typedef std::tuple<float, double> supported_types;

#ifndef log_intel
#define log_intel(x)                   \
  if (IntelKernel<>::IsLogEnabled()) { \
    LOG(INFO) << x;                    \
  }
#endif

static inline Xbyak::Zmm make_zmm(const Xbyak::Xmm &v) {
  return Xbyak::Zmm(v.getIdx());
}
template <int version = 0>
struct IntelKernel {
  static int64_t GetValue() {
    int64_t v = 0;
    const char *label = "DGL_CPU_INTEL_KERNEL_ENABLED";
    const char *ptr = std::getenv(label);
    if (ptr) {
      v = atoll(ptr);
      log_intel(label << "=>" << v);
    }
    return v;
  }

  static int64_t IsEnabled() {
    static int64_t r = IntelKernel<version>::GetValue();
    return r;
  }

  static int IsLogEnabled() {
    static int r = (std::getenv("DGL_CPU_INTEL_KERNEL_LOG")) ? 1 : 0;
    return r;
  }
};

/*!
 * \brief Element-wise addition kernel using Intel AVX512 instructions.
 * \note it uses AVX512.
 */
template <class Op>
class ElemWiseAddUpdate : public Xbyak::CodeGenerator {
 public:
  typedef typename Op::type DType;
  static_assert(
      std::is_base_of<
          std::true_type, utils::has_type<DType, supported_types>>::value,
      "Use case fail dgl::ElemWiseAddUpdate< Operator<DType> > DType is not "
      "supported !");

 protected:
  const Xbyak::Reg64 &r_out_;
  const Xbyak::Reg64 &r_left_;
  const Xbyak::Reg64 &r_right;
  const Xbyak::Reg64 &r_size_;

  /* [functional] Does kernel is applicable on this machine ? */
  bool applicable_;

 public:
  static constexpr int UNIT_SIZE_BYTES = sizeof(DType);
  static constexpr int BITS_IN_BYTES = 8;
  static constexpr int REG_BIT_SIZE = 512;
  static constexpr int UNIT_PER_REG =
      REG_BIT_SIZE / (UNIT_SIZE_BYTES * BITS_IN_BYTES);

  template <
      class TType, class R1, class R2, utils::CheckCmp<TType, float> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovups(r1, r2);
  }
  template <
      class TType, class R1, class R2, utils::CheckCmp<TType, double> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovupd(r1, r2);
  }

  template <
      class TType, class R1, class R2, utils::CheckCmp<TType, float> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }
  template <
      class TType, class R1, class R2, utils::CheckCmp<TType, double> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }

  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, float> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddps(r1, r2, r3);
  }
  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, double> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddpd(r1, r2, r3);
  }

  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, float> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubps(r1, r2, r3);
  }
  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, double> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubpd(r1, r2, r3);
  }

  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, float> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivps(r1, r2, r3);
  }
  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, double> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivpd(r1, r2, r3);
  }

  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, float> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulps(r1, r2, r3);
  }
  template <
      class TType, class R1, class R2, class R3,
      utils::CheckCmp<TType, double> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulpd(r1, r2, r3);
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::CopyLhs, supported_types> =
          true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r9 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_left_ + r9 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r9 * sizeof(IType)], zmm2);
  }
  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::CopyRhs, supported_types> =
          true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r9 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_right + r9 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r9 * sizeof(IType)], zmm2);
  }
  template <class T>
  void loop_pre() {
    alias_load<T>(zmm0, ptr[r_out_ + r9 * sizeof(T)]);
    alias_load<T>(zmm1, ptr[r_left_ + r9 * sizeof(T)]);
    alias_load<T>(zmm2, ptr[r_right + r9 * sizeof(T)]);
  }
  template <class T>
  void loop_post() {
    alias_ADD<T>(zmm2, zmm0, zmm2);
    alias_save<T>(ptr[r_out_ + r9 * sizeof(T)], zmm2);
  }
  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Add, supported_types> =
          true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_ADD<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }
  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Sub, supported_types> =
          true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_SUB<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Div, supported_types> =
          true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_DIV<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Mul, supported_types> =
          true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_MUL<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::CopyLhs, supported_types> =
          true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask, ptr[r_left_ + r9 * sizeof(IType)]);
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::CopyRhs, supported_types> =
          true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask, ptr[r_right + r9 * sizeof(IType)]);
  }

  template <class T>
  void remainder_fetch_LR(const Xbyak::Opmask mask) {
    alias_load<T>(make_zmm(zmm2) | mask, ptr[r_left_ + r9 * sizeof(T)]);
    alias_load<T>(make_zmm(zmm1) | mask, ptr[r_right + r9 * sizeof(T)]);
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Mul, supported_types> =
          true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_MUL<IType>(zmm2, zmm2, zmm1);
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Add, supported_types> =
          true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_ADD<DType>(zmm2, zmm2, zmm1);
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Div, supported_types> =
          true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_DIV<DType>(zmm2, zmm2, zmm1);
  }

  template <
      class Operator,
      utils::Verify<Operator, ::dgl::aten::cpu::op::Sub, supported_types> =
          true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_SUB<DType>(zmm2, zmm2, zmm1);
  }

  ElemWiseAddUpdate()
      : r_out_(rdi),
        r_left_(rsi),
        r_right(rdx),
        r_size_(rcx),
        applicable_(false) {
    static Xbyak::util::Cpu current_cpu;

    /* Default case for all */
    if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
      /* prepare REMAINDER */
      mov(r8, r_size_);
      and_(
          r8,
          UNIT_PER_REG - 1);  // r8_modulo = size/(sizeof(zmm)/sizeof(float))
      xor_(r9, r9);           // reset r9
      cmp(r_size_, UNIT_PER_REG);  // if ( size < 16 ) {  }
      jl("remainder");

      /*  decrease  divident */
      sub(r_size_, r8);  // prepare alignment chunks
      cmp(r_size_, 0);   // do we have any full chunks ?
      jz("remainder");

      L("for_i");
      full_chunk_loop_operations<Op>();
      add(r9, UNIT_PER_REG);  // r9+=sizeof(zmm)/sizeof(float)
      cmp(r_size_, r9);       // more full chunks ?
      jnz("for_i");

      L("remainder");
      cmp(r8, 0);  //  do we have a remainder ?
      jz("done");
      /* prepare a bitmask for k1 */
      mov(rax, 1);
      mov(r_size_, r8);
      sal(rax, cl);
      dec(rax);        // k1= (1 << r8 )-1
      kmovw(k1, eax);  // set bitmask
      alias_load<DType>(
          make_zmm(zmm0) | k1, ptr[r_out_ + r9 * UNIT_SIZE_BYTES]);
      remainder_operations<Op>(k1);
      alias_ADD<DType>(zmm3, zmm2, zmm0);
      alias_save<DType>(
          ptr[r_out_ + r9 * UNIT_SIZE_BYTES], make_zmm(zmm3) | k1);
      L("done");
      applicable_ = true;
      log_intel("AVX512F cpu kernel is ready");
    }
    ret();
  }

  bool applicable() const { return applicable_; }

  template <class... P>
  void run(P... args) {
    ((void (*)(P...))(this)->getCode())(args...);
  }
};

}  // namespace dgl

#endif  // INTEL_CPU_SUPPORT_H_
