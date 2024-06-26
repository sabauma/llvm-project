# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-32: error:
# ERROR-NOT: error:
# CHECK: {evex}	seto	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x40,0xc0]
         {evex}	seto	%al
# CHECK: {evex}	setno	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x41,0xc0]
         {evex}	setno	%al
# CHECK: {evex}	setb	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x42,0xc0]
         {evex}	setb	%al
# CHECK: {evex}	setae	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x43,0xc0]
         {evex}	setae	%al
# CHECK: {evex}	sete	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x44,0xc0]
         {evex}	sete	%al
# CHECK: {evex}	setne	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x45,0xc0]
         {evex}	setne	%al
# CHECK: {evex}	setbe	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x46,0xc0]
         {evex}	setbe	%al
# CHECK: {evex}	seta	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x47,0xc0]
         {evex}	seta	%al
# CHECK: {evex}	sets	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x48,0xc0]
         {evex}	sets	%al
# CHECK: {evex}	setns	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x49,0xc0]
         {evex}	setns	%al
# CHECK: {evex}	setp	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4a,0xc0]
         {evex}	setp	%al
# CHECK: {evex}	setnp	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4b,0xc0]
         {evex}	setnp	%al
# CHECK: {evex}	setl	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4c,0xc0]
         {evex}	setl	%al
# CHECK: {evex}	setge	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4d,0xc0]
         {evex}	setge	%al
# CHECK: {evex}	setle	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4e,0xc0]
         {evex}	setle	%al
# CHECK: {evex}	setg	%al
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4f,0xc0]
         {evex}	setg	%al
# CHECK: {evex}	seto	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x40,0x00]
         {evex}	seto	(%rax)
# CHECK: {evex}	setno	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x41,0x00]
         {evex}	setno	(%rax)
# CHECK: {evex}	setb	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x42,0x00]
         {evex}	setb	(%rax)
# CHECK: {evex}	setae	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x43,0x00]
         {evex}	setae	(%rax)
# CHECK: {evex}	sete	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x44,0x00]
         {evex}	sete	(%rax)
# CHECK: {evex}	setne	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x45,0x00]
         {evex}	setne	(%rax)
# CHECK: {evex}	setbe	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x46,0x00]
         {evex}	setbe	(%rax)
# CHECK: {evex}	seta	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x47,0x00]
         {evex}	seta	(%rax)
# CHECK: {evex}	sets	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x48,0x00]
         {evex}	sets	(%rax)
# CHECK: {evex}	setns	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x49,0x00]
         {evex}	setns	(%rax)
# CHECK: {evex}	setp	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4a,0x00]
         {evex}	setp	(%rax)
# CHECK: {evex}	setnp	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4b,0x00]
         {evex}	setnp	(%rax)
# CHECK: {evex}	setl	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4c,0x00]
         {evex}	setl	(%rax)
# CHECK: {evex}	setge	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4d,0x00]
         {evex}	setge	(%rax)
# CHECK: {evex}	setle	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4e,0x00]
         {evex}	setle	(%rax)
# CHECK: {evex}	setg	(%rax)
# CHECK: encoding: [0x62,0xf4,0x7f,0x08,0x4f,0x00]
         {evex}	setgb	(%rax)
