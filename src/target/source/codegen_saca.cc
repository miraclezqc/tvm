/*
 * Author: zhu qianchao
 * Date: 2021.12.15
 */

/*!
 * \file codegen_saca.cc
 */

#include "codegen_saca.h"


#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../../runtime/thread_storage_scope.h"
#include "literal/saca_common.h"

namespace tvm {
namespace codegen {

CodeGenSACA::CodeGenSACA() { restrict_keyword_ = "__restrict__"; }

void CodeGenSACA::Init() {
  CodeGenC::Init(false); // output_ssa = false
  // vid_global_barrier_state_ = GetUniqueName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = GetUniqueName("__barrier_expect");
  // ICHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenSACA::PrintFuncPrefix() { stream << "__kernel__ void"; }

class ThreadIdxExtractor : public tir::StmtVisitor {
 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" || iv->thread_tag == "threadIdx.x") {
        threadIdx_c_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.y" || iv->thread_tag == "threadIdx.y") {
        threadIdx_r_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 public:
  PrimExpr threadIdx_c_ext = Integer(1);
  PrimExpr threadIdx_r_ext = Integer(1);
};

void CodeGenSACA::PrintExtraAttrs(const PrimFunc& f) {
  ThreadIdxExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext = analyzer.Simplify(extractor.threadIdx_c_ext * extractor.threadIdx_r_ext);
  if (const IntImmNode* const threadIdx_ext_int = threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly return
      return;
    }
    stream << " /*__threads_bounds__(" << threadIdx_ext_int->value << ")*/";
  }
}

std::string CodeGenSACA::Finish() {
  if (enable_fp16_) {
    decl_stream << "#include <crts_fp16.h>\n";
  }

  if (need_crts_h_) {
    decl_stream << "#include <crts.h>\n";
    decl_stream << "#include <slave.h>\n";
    decl_stream << "#include <simd.h>\n";
  }

  decl_stream << "  #define uint unsigned int\n";
  decl_stream << "  #define uchar unsigned char\n";
  decl_stream << "  #define ushort unsigned short\n";
  decl_stream << "  #define int64_t long long\n";
  decl_stream << "  #define uint64_t unsigned long long\n\n";

  return CodeGenC::Finish();
}

void CodeGenSACA::VisitStmt_(const tir::ForNode* op) {
  ICHECK(is_const_int(op->min, 0));
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n"; // todozqc not sure 
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenSACA::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenSACA::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        if (t.is_scalar()) {
          os << "half";
        } else if (lanes <= 8) {
          // Emit CUDA code to access fp16 vector elements.
          //
          // half4 is stored as uint2
          //
          // h4.x is emitted as *(half2*)(&(u2.x)).x
          // h4.y is emitted as *(half2*)(&(u2.x)).y
          // h4.z is emitted as *(half2*)(&(u2.y)).x
          // h4.w is emitted as *(half2*)(&(u2.y)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
          os << "uint" << lanes / 2;
        } else {
          fail = true;
        }
        break;
      case 32:
        if (lanes <= 4) {
          os << "float";
        } else if (lanes <= 8) {
          // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
          //
          // float8 is stored as ulonglong4
          //
          // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
          // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for float type with lanes > 4";
          os << "ulonglong" << lanes / 2;
        } else {
          fail = true;
        }
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16)) return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
      case 1: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          os << "int8_t";
          return;
        } else if (t.lanes() == 16) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 32) {
          os << "int";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 4: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 4) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 8) {
          // directly 8 4-bit int in integer.
          os << "int";
          return;
        } else if (t.lanes() == 16) {
          os << "int2";
          return;
        } else if (t.lanes() == 32) {
          os << "int4";
          return;
        } else if (t.lanes() == 64) {
          os << "int8";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 8: {
        if (t.lanes() == 4) {
          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          os << "int2";
          return;
        } else if (t.lanes() == 16) {
          os << "int4";
          return;
        } else if (!t.is_uint() && t.is_scalar()) {
          os << "signed char";
          break;
        } else {
          os << "char";
          break;
        }
      }
      case 16: {
        if (t.is_scalar()) {
          os << "short";
        } else if (t.lanes() <= 4) {
          os << "short" << lanes;
        } else if (t.lanes() <= 8) {
          // Emit CUDA code to access int16 vector elements.
          //
          // short4 is stored as int2
          //
          // s4.x is emitted as *(short2*)(&(i2.x)).x
          // s4.y is emitted as *(short2*)(&(i2.x)).y
          // s4.z is emitted as *(short2*)(&(i2.y)).x
          // s4.w is emitted as *(short2*)(&(i2.y)).y
          //
          ICHECK_EQ(t.lanes() % 2, 0) << "only support even lane for shorT type with lanes > 4";
          os << "int" << t.lanes() / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 32: {
        if (t.is_scalar()) {
          os << "int";
        } else if (t.lanes() <= 4) {
          os << "int" << t.lanes();
        } else if (t.lanes() <= 8) {
          // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
          //
          // int8 is stored as longlong4
          //
          // i8.v1 is emitted as *(int2*)(&(l4.x)).x
          // i8.v2 is emitted as *(int2*)(&(l4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for int32 type with lanes > 4";
          os << "longlong" << lanes / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 64: {
        if (t.is_scalar()) {
          os << "int64_t";
        } else if (t.lanes() == 2) {
          os << "longlong2";
        } else if (t.lanes() == 3) {
          os << "longlong3";
        } else if (t.lanes() == 4) {
          os << "longlong4";
        }
        return;
      }
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenSACA::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                   std::ostream& os) {  // NOLINT(*)
  // Delcare the result.
  std::string sret = GetUniqueName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenSACA::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                   std::ostream& os) {  // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if (t.is_float16()) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenSACA::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                    const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2] << " = "
           << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenSACA::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "CRTS_ssync_array();\n";
  } else if (sync == "global") {
    LOG(FATAL) << "Global barrier is not needed for SACA" << op->dtype << "\n";
  }
}

void CodeGenSACA::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting SACA. You must pass "
                                "all global arrays as input instead";
  if (scope == "shared") {
    // os << "__shared__ ";
    os << " DMA something "; // todozqc 
   } else if (scope == "shared.dyn") {
     os << "extern thread_local ";  // todozqc 
  }
}

void CodeGenSACA::VisitExpr_(const CastNode* op, std::ostream& os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar()) return CodeGenC::VisitExpr_(op, os);

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = GetUniqueName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.element_of(), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenSACA::PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                  bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_vector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = GetUniqueName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0) scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
  }
}

void CodeGenSACA::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (auto* ptr_op = op->op.as<OpNode>()) {
    Op call_op = GetRef<Op>(ptr_op);
    // This is only for backward compatibility with __shfl_{up/down}.
    // A macro will be used to replace *_sync calls to legacy ones.
  }

  // if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
  //   need_mma_h_ = true;
  //   ICHECK_EQ(op->args.size(), 8U);
  //   os << "crts::load_matrix_sync(";
  //   this->PrintExpr(op->args[0], os);
  // } else {
    CodeGenC::VisitExpr_(op, os);
  //}
}

void CodeGenSACA::VisitStmt_(const AttrStmtNode* op) {
  CodeGenC::VisitStmt_(op);
}

void CodeGenSACA::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode* buffer = op->buffer_var.as<VarNode>();
  // if (scope.find("smth") == 0) {
  //   continue; // todozqc
  // } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
  // }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    int32_t constant_size = op->constant_allocation_size();
    ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now(SACA)";

    if ((op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
         op->dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / op->dtype.bits());
    }
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenSACA::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (CRTS_tid == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenSACA::VisitExpr_(const RampNode* op, std::ostream& os) {
  os << "(make_int" << op->lanes << "(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "))";
}

void CodeGenSACA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 && op->lanes == 4) {
    // make_int8x4
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (op->dtype.is_uint()) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < op->lanes / 2; ++i) {
      if (i != 0) os << ", ";
      os << "__pack_half2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xF;

    if (op->lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) | (v << 4) | v;
      if (op->lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (op->lanes == 16 || op->lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < op->lanes / 8; ++i) {
          if (i != 0) os << ", ";
          if (op->dtype.is_uint()) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenSACA::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  std::vector<std::string> to_shuffle(op->vectors.size());
  for (int i = 0, e = op->vectors.size(); i < e; ++i) {
    ICHECK(op->vectors[i].dtype().lanes() == 1) << "Only scalars can be shuffled!";//todozqc
    to_shuffle[i] = PrintExpr(op->vectors[i]);
  }
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0, e = op->indices.size(); i < e; ++i) {
    const int64_t* val = as_const_int(op->indices[i]);
    ICHECK(val && *val >= 0 && (int)*val < (int)to_shuffle.size());
    if (i != 0) os << ", ";
    os << to_shuffle[*val];
  }
  os << ')';
}

void CodeGenSACA::VisitExpr_(const SelectNode* op, std::ostream& os) {
  // Non-vector cases.
  if (!op->dtype.is_vector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  ICHECK(op->false_value->dtype == op->dtype && op->true_value->dtype == op->dtype &&
         op->dtype.lanes() == op->condition.dtype().lanes());

  std::string r_var = GetUniqueName("_");
  this->PrintIndent();
  this->PrintType(op->dtype, stream);
  stream << ' ' << r_var << ";\n";
  {
    std::string c_var = SSAGetID(PrintExpr(op->condition), op->dtype);
    std::string t_var = SSAGetID(PrintExpr(op->true_value), op->dtype);
    std::string f_var = SSAGetID(PrintExpr(op->false_value), op->dtype);

    // The condition is stored as an ushort vector.
    int lanes = op->dtype.lanes();
    DataType memory_ty(DataType::TypeCode::kUInt, 16, lanes);

    for (int i = 0; i < lanes; ++i) {
      std::ostringstream item;
      item << "(bool(";
      PrintVecElemLoad(c_var, memory_ty, i, item);
      item << ")?";
      PrintVecElemLoad(t_var, op->dtype, i, item);
      item << ':';
      PrintVecElemLoad(f_var, op->dtype, i, item);
      item << ')';
      PrintVecElemStore(r_var, op->dtype, i, item.str());
    }
  }
  os << r_var;
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenSACA* p) {  // NOLINT(*)
  // Type code is kFloat
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << ((op->dtype.bits() == 32) ? "INF_F" : "INF");
      } else if (std::isnan(op->value)) {
        temp << ((op->dtype.bits() == 32) ? "NAN_F" : "NAN");
      } else {
        temp << std::scientific << op->value;
        if (op->dtype.bits() == 32) temp << 'f';
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half_rn";
      os << '(' << std::scientific << op->value << 'f' << ')';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenSACA::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}


void CodeGenSACA::HandleVolatileLoads(const std::string& value, const LoadNode* op,
                                      std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16()) && IsVolatile(op->buffer_var.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenSACA::PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
                                       std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
      return;
    }
  }

  if (t.is_float16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_half2(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (i == 0) {
    os << "make_";
    PrintType(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

}  // namespace codegen
}  // namespace tvm
