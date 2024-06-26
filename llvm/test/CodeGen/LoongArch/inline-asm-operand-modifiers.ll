; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc --mtriple=loongarch32 -mattr=+d --verify-machineinstrs < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 -mattr=+d --verify-machineinstrs < %s | FileCheck %s

define i32 @modifier_z_zero(i32 %a) nounwind {
; CHECK-LABEL: modifier_z_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    add.w $a0, $a0, $zero
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    ret
  %1 = tail call i32 asm "add.w $0, $1, ${2:z}", "=r,r,ri"(i32 %a, i32 0)
  ret i32 %1
}

define i32 @modifier_z_nonzero(i32 %a) nounwind {
; CHECK-LABEL: modifier_z_nonzero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    addi.w $a0, $a0, 1
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    ret
  %1 = tail call i32 asm "addi.w $0, $1, ${2:z}", "=r,r,ri"(i32 %a, i32 1)
  ret i32 %1
}
