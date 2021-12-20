/*
 * Author: zhu qianchao
 * Date: 2021.12.15
 */

/*!
 * \file saca_module.h
 * \brief Execution handling of SACA kernels
 */
#ifndef TVM_RUNTIME_SACA_MODULE_H_
#define TVM_RUNTIME_SACA_MODULE_H_

#include <tvm/runtime/module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of Sunway-manycore supported in SACAModule */
static constexpr const int MaxThreads = 64;

/*!
 * \brief create a saca module from data.
 *
 * \param data The module data, can be ptx, cubin
 * \param fmt The format of the data, can be "ptx", "cubin"
 * \param fmap The map function information map of each function.
 * \param cuda_source Optional, cuda source file
 */
Module SACAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_SACA_MODULE_H_
