/*
 * Author: zhu qianchao
 * Date: 2021.12.15
 */

/*!
 * \file saca_common.h
 * \brief common function/instruction for thw SACA.
 */
#ifndef TVM_TARGET_SOURCE_LITERAL_SACA_COMMON_H_
#define TVM_TARGET_SOURCE_LITERAL_SACA_COMMON_H_

static constexpr const char* __const_info_def = R"(
#define NUM_CORES 64
#define COL_CORES 8
#define ROW_CORES 8
#define ELL_SIZE 27
#define ALIGN_SIZE 512
)";

static constexpr const char* _rld_function = R"(
#define addressof(type,p,tid) ((type*)((unsigned long)(p) | (((unsigned long)tid) << 20) | (1ul << 45)))

#define getval(type,p,tid) *addressof(type,p,tid)
)";

#endif  // TVM_TARGET_SOURCE_LITERAL_CUDA_HALF_T_H_
