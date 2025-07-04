; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i686-linux-gnu -fast-isel  | FileCheck %s --check-prefixes=X86,FASTISEL-X86
; RUN: llc < %s -mtriple=x86_64-linux-gnu -fast-isel  | FileCheck %s --check-prefixes=X64,FASTISEL-X64
; RUN: llc < %s -mtriple=i686-linux-gnu -global-isel=0 -fast-isel=0  | FileCheck %s --check-prefixes=X86,SDAG-X86
; RUN: llc < %s -mtriple=x86_64-linux-gnu -global-isel=0 -fast-isel=0  | FileCheck %s --check-prefixes=X64,SDAG-X64
; TODO: The below RUN line will fails GISEL selection and will fallback to DAG selection due to lack of support for loads/stores in i686 mode, support is expected soon enough, for this reason the llvm/test/CodeGen/X86/GlobalISel/llvm.sincos.mir test is added for now because of the lack of support for i686 in GlobalISel.
; RUN: llc < %s -mtriple=i686-linux-gnu -global-isel=1 -global-isel-abort=2 | FileCheck %s --check-prefixes=GISEL-X86
; RUN: llc < %s -mtriple=x86_64-linux-gnu -global-isel=1 -global-isel-abort=1 | FileCheck %s --check-prefixes=GISEL-X64

define { float, float } @test_sincos_f32(float %Val) nounwind {
; X86-LABEL: test_sincos_f32:
; X86:       # %bb.0:
; X86-NEXT:    subl $28, %esp
; X86-NEXT:    flds {{[0-9]+}}(%esp)
; X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-NEXT:    fstps (%esp)
; X86-NEXT:    calll sincosf
; X86-NEXT:    flds {{[0-9]+}}(%esp)
; X86-NEXT:    flds {{[0-9]+}}(%esp)
; X86-NEXT:    addl $28, %esp
; X86-NEXT:    retl
;
; X64-LABEL: test_sincos_f32:
; X64:       # %bb.0:
; X64-NEXT:    pushq %rax
; X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdi
; X64-NEXT:    movq %rsp, %rsi
; X64-NEXT:    callq sincosf@PLT
; X64-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; X64-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X64-NEXT:    popq %rax
; X64-NEXT:    retq
;
; GISEL-X86-LABEL: test_sincos_f32:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $28, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %edx
; GISEL-X86-NEXT:    movl %eax, (%esp)
; GISEL-X86-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    movl %edx, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    calll sincosf
; GISEL-X86-NEXT:    flds {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    flds {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fxch %st(1)
; GISEL-X86-NEXT:    addl $28, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: test_sincos_f32:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdi
; GISEL-X64-NEXT:    movq %rsp, %rsi
; GISEL-X64-NEXT:    callq sincosf
; GISEL-X64-NEXT:    movl {{[0-9]+}}(%rsp), %eax
; GISEL-X64-NEXT:    movl (%rsp), %ecx
; GISEL-X64-NEXT:    movd %eax, %xmm0
; GISEL-X64-NEXT:    movd %ecx, %xmm1
; GISEL-X64-NEXT:    popq %rax
; GISEL-X64-NEXT:    retq
  %res = call { float, float } @llvm.sincos.f32(float %Val)
  ret { float, float } %res
}

define { double, double } @test_sincos_f64(double %Val) nounwind  {
; X86-LABEL: test_sincos_f64:
; X86:       # %bb.0:
; X86-NEXT:    subl $44, %esp
; X86-NEXT:    fldl {{[0-9]+}}(%esp)
; X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-NEXT:    fstpl (%esp)
; X86-NEXT:    calll sincos
; X86-NEXT:    fldl {{[0-9]+}}(%esp)
; X86-NEXT:    fldl {{[0-9]+}}(%esp)
; X86-NEXT:    addl $44, %esp
; X86-NEXT:    retl
;
; X64-LABEL: test_sincos_f64:
; X64:       # %bb.0:
; X64-NEXT:    subq $24, %rsp
; X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdi
; X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rsi
; X64-NEXT:    callq sincos@PLT
; X64-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X64-NEXT:    movsd {{.*#+}} xmm1 = mem[0],zero
; X64-NEXT:    addq $24, %rsp
; X64-NEXT:    retq
;
; GISEL-X86-LABEL: test_sincos_f64:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $44, %esp
; GISEL-X86-NEXT:    fldl {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpl (%esp)
; GISEL-X86-NEXT:    calll sincos
; GISEL-X86-NEXT:    fldl {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fldl {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    addl $44, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: test_sincos_f64:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdi
; GISEL-X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rsi
; GISEL-X64-NEXT:    callq sincos
; GISEL-X64-NEXT:    movq {{[0-9]+}}(%rsp), %rax
; GISEL-X64-NEXT:    movq {{[0-9]+}}(%rsp), %rcx
; GISEL-X64-NEXT:    movq %rax, %xmm0
; GISEL-X64-NEXT:    movq %rcx, %xmm1
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %res = call { double, double } @llvm.sincos.f64(double %Val)
  ret { double, double } %res
}

define { x86_fp80, x86_fp80 } @test_sincos_f80(x86_fp80 %Val) nounwind {
; X86-LABEL: test_sincos_f80:
; X86:       # %bb.0:
; X86-NEXT:    subl $44, %esp
; X86-NEXT:    fldt {{[0-9]+}}(%esp)
; X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; X86-NEXT:    fstpt (%esp)
; X86-NEXT:    calll sincosl
; X86-NEXT:    fldt {{[0-9]+}}(%esp)
; X86-NEXT:    fldt {{[0-9]+}}(%esp)
; X86-NEXT:    addl $44, %esp
; X86-NEXT:    retl
;
; X64-LABEL: test_sincos_f80:
; X64:       # %bb.0:
; X64-NEXT:    subq $56, %rsp
; X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; X64-NEXT:    fstpt (%rsp)
; X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdi
; X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rsi
; X64-NEXT:    callq sincosl@PLT
; X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; X64-NEXT:    addq $56, %rsp
; X64-NEXT:    retq
;
; GISEL-X86-LABEL: test_sincos_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $60, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; GISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %ecx
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    movl %ecx, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    calll sincosl
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fxch %st(1)
; GISEL-X86-NEXT:    addl $60, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: test_sincos_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $56, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rdi
; GISEL-X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rsi
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq sincosl
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fxch %st(1)
; GISEL-X64-NEXT:    addq $56, %rsp
; GISEL-X64-NEXT:    retq
  %res = call { x86_fp80, x86_fp80 } @llvm.sincos.f80(x86_fp80 %Val)
  ret { x86_fp80, x86_fp80 } %res
}

declare void @foo(ptr, ptr)

define void @can_fold_with_call_in_chain(float %x, ptr noalias %a, ptr noalias %b) nounwind {
; FASTISEL-X86-LABEL: can_fold_with_call_in_chain:
; FASTISEL-X86:       # %bb.0: # %entry
; FASTISEL-X86-NEXT:    pushl %edi
; FASTISEL-X86-NEXT:    pushl %esi
; FASTISEL-X86-NEXT:    subl $20, %esp
; FASTISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %esi
; FASTISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %edi
; FASTISEL-X86-NEXT:    flds {{[0-9]+}}(%esp)
; FASTISEL-X86-NEXT:    fstps {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Spill
; FASTISEL-X86-NEXT:    movl %esi, {{[0-9]+}}(%esp)
; FASTISEL-X86-NEXT:    movl %edi, (%esp)
; FASTISEL-X86-NEXT:    calll foo@PLT
; FASTISEL-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; FASTISEL-X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; FASTISEL-X86-NEXT:    movl %edi, {{[0-9]+}}(%esp)
; FASTISEL-X86-NEXT:    flds {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Reload
; FASTISEL-X86-NEXT:    fstps (%esp)
; FASTISEL-X86-NEXT:    calll sincosf
; FASTISEL-X86-NEXT:    flds {{[0-9]+}}(%esp)
; FASTISEL-X86-NEXT:    fstps (%esi)
; FASTISEL-X86-NEXT:    addl $20, %esp
; FASTISEL-X86-NEXT:    popl %esi
; FASTISEL-X86-NEXT:    popl %edi
; FASTISEL-X86-NEXT:    retl
;
; FASTISEL-X64-LABEL: can_fold_with_call_in_chain:
; FASTISEL-X64:       # %bb.0: # %entry
; FASTISEL-X64-NEXT:    pushq %r14
; FASTISEL-X64-NEXT:    pushq %rbx
; FASTISEL-X64-NEXT:    pushq %rax
; FASTISEL-X64-NEXT:    movq %rsi, %rbx
; FASTISEL-X64-NEXT:    movq %rdi, %r14
; FASTISEL-X64-NEXT:    movss %xmm0, (%rsp) # 4-byte Spill
; FASTISEL-X64-NEXT:    callq sinf@PLT
; FASTISEL-X64-NEXT:    movss %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 4-byte Spill
; FASTISEL-X64-NEXT:    movss (%rsp), %xmm0 # 4-byte Reload
; FASTISEL-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; FASTISEL-X64-NEXT:    callq cosf@PLT
; FASTISEL-X64-NEXT:    movss %xmm0, (%rsp) # 4-byte Spill
; FASTISEL-X64-NEXT:    movq %r14, %rdi
; FASTISEL-X64-NEXT:    movq %rbx, %rsi
; FASTISEL-X64-NEXT:    callq foo@PLT
; FASTISEL-X64-NEXT:    movss {{[-0-9]+}}(%r{{[sb]}}p), %xmm0 # 4-byte Reload
; FASTISEL-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; FASTISEL-X64-NEXT:    movss %xmm0, (%r14)
; FASTISEL-X64-NEXT:    movss (%rsp), %xmm0 # 4-byte Reload
; FASTISEL-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; FASTISEL-X64-NEXT:    movss %xmm0, (%rbx)
; FASTISEL-X64-NEXT:    addq $8, %rsp
; FASTISEL-X64-NEXT:    popq %rbx
; FASTISEL-X64-NEXT:    popq %r14
; FASTISEL-X64-NEXT:    retq
;
; SDAG-X86-LABEL: can_fold_with_call_in_chain:
; SDAG-X86:       # %bb.0: # %entry
; SDAG-X86-NEXT:    pushl %edi
; SDAG-X86-NEXT:    pushl %esi
; SDAG-X86-NEXT:    subl $20, %esp
; SDAG-X86-NEXT:    flds {{[0-9]+}}(%esp)
; SDAG-X86-NEXT:    fstps {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Spill
; SDAG-X86-NEXT:    movl {{[0-9]+}}(%esp), %edi
; SDAG-X86-NEXT:    movl {{[0-9]+}}(%esp), %esi
; SDAG-X86-NEXT:    movl %esi, {{[0-9]+}}(%esp)
; SDAG-X86-NEXT:    movl %edi, (%esp)
; SDAG-X86-NEXT:    calll foo@PLT
; SDAG-X86-NEXT:    leal {{[0-9]+}}(%esp), %eax
; SDAG-X86-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; SDAG-X86-NEXT:    movl %edi, {{[0-9]+}}(%esp)
; SDAG-X86-NEXT:    flds {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Reload
; SDAG-X86-NEXT:    fstps (%esp)
; SDAG-X86-NEXT:    calll sincosf
; SDAG-X86-NEXT:    flds {{[0-9]+}}(%esp)
; SDAG-X86-NEXT:    fstps (%esi)
; SDAG-X86-NEXT:    addl $20, %esp
; SDAG-X86-NEXT:    popl %esi
; SDAG-X86-NEXT:    popl %edi
; SDAG-X86-NEXT:    retl
;
; SDAG-X64-LABEL: can_fold_with_call_in_chain:
; SDAG-X64:       # %bb.0: # %entry
; SDAG-X64-NEXT:    pushq %r14
; SDAG-X64-NEXT:    pushq %rbx
; SDAG-X64-NEXT:    pushq %rax
; SDAG-X64-NEXT:    movq %rsi, %rbx
; SDAG-X64-NEXT:    movq %rdi, %r14
; SDAG-X64-NEXT:    movss %xmm0, (%rsp) # 4-byte Spill
; SDAG-X64-NEXT:    callq foo@PLT
; SDAG-X64-NEXT:    leaq {{[0-9]+}}(%rsp), %rsi
; SDAG-X64-NEXT:    movss (%rsp), %xmm0 # 4-byte Reload
; SDAG-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; SDAG-X64-NEXT:    movq %r14, %rdi
; SDAG-X64-NEXT:    callq sincosf@PLT
; SDAG-X64-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SDAG-X64-NEXT:    movss %xmm0, (%rbx)
; SDAG-X64-NEXT:    addq $8, %rsp
; SDAG-X64-NEXT:    popq %rbx
; SDAG-X64-NEXT:    popq %r14
; SDAG-X64-NEXT:    retq
;
; GISEL-X86-LABEL: can_fold_with_call_in_chain:
; GISEL-X86:       # %bb.0: # %entry
; GISEL-X86-NEXT:    pushl %ebx
; GISEL-X86-NEXT:    pushl %edi
; GISEL-X86-NEXT:    pushl %esi
; GISEL-X86-NEXT:    subl $16, %esp
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %ebx
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %esi
; GISEL-X86-NEXT:    movl {{[0-9]+}}(%esp), %edi
; GISEL-X86-NEXT:    movl %ebx, (%esp)
; GISEL-X86-NEXT:    calll sinf
; GISEL-X86-NEXT:    fstps {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Spill
; GISEL-X86-NEXT:    movl %ebx, (%esp)
; GISEL-X86-NEXT:    calll cosf
; GISEL-X86-NEXT:    fstps {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Spill
; GISEL-X86-NEXT:    movl %esi, (%esp)
; GISEL-X86-NEXT:    movl %edi, {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    calll foo
; GISEL-X86-NEXT:    flds {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Reload
; GISEL-X86-NEXT:    fstps (%esi)
; GISEL-X86-NEXT:    flds {{[-0-9]+}}(%e{{[sb]}}p) # 4-byte Folded Reload
; GISEL-X86-NEXT:    fstps (%edi)
; GISEL-X86-NEXT:    addl $16, %esp
; GISEL-X86-NEXT:    popl %esi
; GISEL-X86-NEXT:    popl %edi
; GISEL-X86-NEXT:    popl %ebx
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: can_fold_with_call_in_chain:
; GISEL-X64:       # %bb.0: # %entry
; GISEL-X64-NEXT:    pushq %r14
; GISEL-X64-NEXT:    pushq %rbx
; GISEL-X64-NEXT:    pushq %rax
; GISEL-X64-NEXT:    movss %xmm0, (%rsp) # 4-byte Spill
; GISEL-X64-NEXT:    movq %rdi, %rbx
; GISEL-X64-NEXT:    movq %rsi, %r14
; GISEL-X64-NEXT:    callq sinf
; GISEL-X64-NEXT:    movss %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 4-byte Spill
; GISEL-X64-NEXT:    movss (%rsp), %xmm0 # 4-byte Reload
; GISEL-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; GISEL-X64-NEXT:    callq cosf
; GISEL-X64-NEXT:    movss %xmm0, (%rsp) # 4-byte Spill
; GISEL-X64-NEXT:    movq %rbx, %rdi
; GISEL-X64-NEXT:    movq %r14, %rsi
; GISEL-X64-NEXT:    callq foo
; GISEL-X64-NEXT:    movd {{[-0-9]+}}(%r{{[sb]}}p), %xmm0 # 4-byte Folded Reload
; GISEL-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; GISEL-X64-NEXT:    movd %xmm0, %eax
; GISEL-X64-NEXT:    movl %eax, (%rbx)
; GISEL-X64-NEXT:    movd (%rsp), %xmm0 # 4-byte Folded Reload
; GISEL-X64-NEXT:    # xmm0 = mem[0],zero,zero,zero
; GISEL-X64-NEXT:    movd %xmm0, %eax
; GISEL-X64-NEXT:    movl %eax, (%r14)
; GISEL-X64-NEXT:    addq $8, %rsp
; GISEL-X64-NEXT:    popq %rbx
; GISEL-X64-NEXT:    popq %r14
; GISEL-X64-NEXT:    retq
entry:
  %sin = tail call float @llvm.sin.f32(float %x)
  %cos = tail call float @llvm.cos.f32(float %x)
  call void @foo(ptr %a, ptr %b)
  store float %sin, ptr %a, align 4
  store float %cos, ptr %b, align 4
  ret void
}
