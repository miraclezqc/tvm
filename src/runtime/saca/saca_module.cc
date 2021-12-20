/*
 * Author: zhu qianchao
 * Date: 2021.12.15
 */

/*!
 * \file saca_module.cc
 */
#include "saca_module.h"

#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"

namespace tvm {
namespace runtime {

class SACAModuleNode : public runtime::ModuleNode {
 public:
  explicit SACAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string saca_source)
      : data_(data), fmt_(fmt), fmap_(fmap), saca_source_(saca_source) {
    // std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~SACAModuleNode() {}

  const char* type_key() const final { return "saca"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "sa") {
      ICHECK_NE(saca_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, saca_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (saca_source_.length() != 0) {
      return saca_source_;
    } else {
      if (fmt_ == "mid_code") return data_;
      return "";
    }
  }

 private:
  // the mid_code data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The saca source.
  std::string saca_source_;
  // the internal modules per Sunway Core, to be lazily initialized.
  // std::array<SACAmodule, 32> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class SACAWrappedFunc {
 public:
  // initialize the SACA function.
  void Init(SACAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    // std::fill(fcache_.begin(), fcache_.end(), nullptr);
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments 
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    // todozqc
    // int device_id;
    // CUDA_CALL(cudaGetDevice(&device_id));
    // if (fcache_[device_id] == nullptr) {
    //   fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    // }
    // CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    // ThreadWorkLoad wl = launch_param_config_.Extract(args);
    // CUresult result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
    //                                  wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
    //                                  wl.block_dim(2), wl.dyn_shmem_size, strm, void_args, nullptr);
    // if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    //   const char* msg;
    //   cuGetErrorName(result, &msg);
    //   std::ostringstream os;
    //   os << "CUDALaunch Error: " << msg << "\n"
    //      << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
    //      << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
    //      << ")\n";
    //   std::string cuda = m_->GetSource("");
    //   if (cuda.length() != 0) {
    //     os << "// func_name=" << func_name_ << "\n"
    //        << "// CUDA Source\n"
    //        << "// -----------\n"
    //        << cuda;
    //   }
    //   LOG(FATAL) << os.str();
    // }
  }

 private:
  // internal module
  SACAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  // mutable std::array<SACAfunction, 32> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};


PackedFunc SACAModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Sunway CPE function do not have main func";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  SACAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module SACAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string saca_source) {
  auto n = make_object<SACAModuleNode>(data, fmt, fmap, saca_source);
  return Module(n);
}

// Load module from module.
Module SACAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SACAModuleCreate(data, fmt, fmap, std::string());
}

Module SACAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SACAModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_saca").set_body_typed(SACAModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
