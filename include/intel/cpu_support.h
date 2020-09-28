/*!
 *  Copyright (c) 2019 by Contributors
 * \file intel/cpu_support.h
 * \brief Intel CPU support
 */
#ifndef INTEL_CPU_SUPPORT_H_
#define INTEL_CPU_SUPPORT_H_
#include <memory>
#include <type_traits>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
namespace intel {
#define log_intel(x) if (IntelKernel<T>::log_enabled()) { std::cout << x << std::endl; }
#ifndef log_intel
#define log_intel(x)
#endif
template<class T>
using Uptr = std::unique_ptr<T>;

template<class T>
struct IntelKernel {
      static int64_t getValue() {
        int64_t v = 0;
        const char *label = "DGL_CPU_INTEL_KERNEL_ENABLED";
        const char *ptr = std::getenv(label);
        if (ptr) {
          v = atoll(ptr);
          log_intel(label << "=>" << v);
        }
        return v;
      }

      static int64_t enabled() {
        static int64_t r = IntelKernel<T>::getValue();
        return r;
      }

      static int log_enabled() {
        static int r = (std::getenv("DGL_CPU_INTEL_KERNEL_LOG")) ? 1 : 0;
        return r;
      }
};

template <class T, int has_specialization = 0>
class elem_wise_add_update : public Xbyak::CodeGenerator {
    typedef elem_wise_add_update<T> self;
    /* [performance] skip static check*/
    int64_t size;

    /* [functional] Does kernel is applicable on this machine ? */
    bool applicable;

    /* [performance-functional] use kernel when input size >= min_size */
    int64_t min_size;

 public:
     void prolog() {
        push(r8);
        push(r9);
     }

      void epilog() {
        pop(r9);
        pop(r8);
      }
    explicit elem_wise_add_update(int64_t _size) :
     size(_size), applicable(false), min_size(IntelKernel<T>::enabled()) {
      static Xbyak::util::Cpu current_cpu;
      /* input => { src=RSI , dst=RDI , size=RDX } */

      /* Default case for all */
      if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
        prolog();
        /* prepare REMAINDER */
        mov(r8, rdx);  // rdx => size
        and_(r8, 0xf);  // r8_modulo = size/(sizeof(zmm)/sizeof(float))
        xor_(r9, r9);  // reset r9
        cmp(rdx, 0x10);  // if ( size < 16 ) {  }
        jl("remainder");

        /*  decrease  divident */
        sub(rdx, r8);  // prepare alignment chunks
        cmp(rdx, 0);  // do we have any full chunks ?
        jz("remainder");

        L("for_i");
        /* load first part of mem to zmm0*/
        vmovups(zmm0, ptr[rdi + r9 * 4]);
        /* load second part of mem to zmm1*/
        vmovups(zmm1, ptr[rsi + r9 * 4]);
        /* zmm2 = zmm0 + zmm1 */
        vaddps(zmm2, zmm0, zmm1);
        /* save output to dst*/
        vmovups(ptr[rdi + r9 * 4], zmm2);
        add(r9, 16);  // r9+=sizeof(zmm)/sizeof(float)
        cmp(rdx, r9);  // more full chunks ?
        jnz("for_i");

        L("remainder");
        cmp(r8, 0);  //  do we have a remainder ?
        jz("done");
        xor_(rax, rax);
        /* prepare a bitmask for k1 */
        mov(rax, 1);
        mov(rcx, r8);
        sal(rax, cl);
        dec(rax);  // k1= (1 << r8 )-1
        kmovw(k1, eax);  // set bitmask
        /* same logic as above but with mask */
        /* vmovups zmm0{k1},ZMMWORD PTR [rdi+r9*4] */
        const uint8_t ptr_3[7] = {0x62, 0xb1, 0x7c, 0x49, 0x10, 0x04, 0x8f};
        db(ptr_3, sizeof(ptr_3) / sizeof(uint8_t));
        /* vmovups zmm1{k1},ZMMWORD PTR [rsi+r9*4] */
        const uint8_t ptr_2[7] = {0x62, 0xb1, 0x7c, 0x49, 0x10, 0x0c, 0x8e};
        db(ptr_2, sizeof(ptr_2) / sizeof(uint8_t));
        /*  zmm2 = zmm0 + zmm1 */
        vaddps(zmm2, zmm0, zmm1);
        /* vmovups ZMMWORD PTR [rdi+r9*4]{k1},zmm2  */
        const uint8_t ptr_1[7] = {0x62, 0xB1, 0x7C, 0x49, 0x11, 0x14, 0x8F};
        db(ptr_1, sizeof(ptr_1) / sizeof(uint8_t));
        L("done");

        epilog();
        applicable = true;
        log_intel("*** AVX512F cpu kernel is ready for size=" << size << " ***");
      }
      ret();
    }

    bool is_applicable() const {
      return applicable;
    }

    /*[functional] generate a new kernel for this size*/
    template<class R>
    typename std::enable_if<has_specialization, R>::type
    require_new_instance(int64_t _size) {
      return  (_size != size) && (_size >= min_size);
    }

    /*[performance] if specialization doesn't exist skip allocation use always default case */
    template <class R>
    typename std::enable_if<!has_specialization, R>::type
    require_new_instance(int64_t _size) {
      return false;
    }

     template<class ... P>
     void run(P ... args) {
         ((void(*)(P...))(this)->getCode())(args...);
     }
};

}  // namespace intel

#endif  // INTEL_CPU_SUPPORT_H_
