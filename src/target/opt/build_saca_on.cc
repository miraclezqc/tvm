/*
 * Author: zhu qianchao
 * Date: 2021.12.15
 */


/*!
 *  Build saca modules from source.
 *  requires saca to be available.
 *
 * \file build_saca.cc
 */
#if defined(__linux__)
#include <sys/stat.h>
#endif

#include <cstdlib>

#include "../../runtime/saca/saca_module.h"
#include "../build_common.h"
#include "../source/codegen_saca.h"

namespace tvm {
namespace codegen {


std::string FindSACAIncludePath() {
  // must linux
  const std::string delimiter = "/";

  std::string saca_include_path;
  const char* saca_path_env = std::getenv("SACA_PATH");
  if (saca_path_env != nullptr) {
    saca_include_path += saca_path_env;
    saca_include_path += delimiter + "include";
    return saca_include_path;
  }

#if defined(__linux__)
  struct stat st; 
  saca_include_path = "/usr/local/sw9gcc/include/"; //todozqc
  if (stat(saca_include_path.c_str(), &st) == 0) {
    return saca_include_path;
  }

#endif
  LOG(FATAL) << "Cannot find saca include path."
             << "SACA_PATH is not set or SACA is not installed in the default installation path."
             << "In other than linux, it is necessary to set SACA_PATH.";
  return saca_include_path;
}

std::string SWGCCCompile(const std::string& code, bool include_path = false) {
  std::string mid_code;
  //todozqc
  return mid_code;
}

runtime::Module BuildSACA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenSACA cg;
  cg.Init();

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSACA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSACA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  // if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
  //   code = (*f)(code).operator std::string();
  // }
  std::string fmt = "mid_code";//todozqc
  std::string ptx;
  // if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
  //   ptx = (*f)(code).operator std::string();
  //   // Dirty matching to check PTX vs cubin.
  //   // TODO(tqchen) more reliable checks
  //   if (ptx[0] != '/') fmt = "cubin";
  // } else {
  //   ptx = SWGCCCompile(code, cg.need_include_path());
  // }
  return SACAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.saca").set_body_typed(BuildSACA);
}  // namespace codegen
}  // namespace tvm
