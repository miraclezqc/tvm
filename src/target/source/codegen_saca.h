/*
 * Author: zhu qianchao
 * Date: 2021.12.15
 */

/*!
 * \file codegen_saca.h
 * \brief Utility to generate saca code
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_SACA_H_
#define TVM_TARGET_SOURCE_CODEGEN_SACA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenSACA final : public CodeGenC {
 public:
  CodeGenSACA();
  void Init();
  std::string Finish();
  bool need_include_path() {
    return (enable_fp16_ || need_crts_h_);
  }
  // override behavior
  void PrintFuncPrefix() final;
  void PrintExtraAttrs(const PrimFunc& f) final;
  void VisitStmt_(const ForNode* op) final;
  void PrintStorageSync(const CallNode* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                        std::ostream& os) final;       // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, DataType t, int i, const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) final;
  // overload visitor
  void VisitExpr_(const RampNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;
  void VisitExpr_(const CallNode* op, std::ostream& os) final;
  void VisitExpr_(const CastNode* op, std::ostream& os) final;
  void VisitStmt_(const EvaluateNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

 protected:
  void PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                       bool skip_first_arg, std::ostream& os) final;  // NOLINT(*)

 private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string& value, const LoadNode* op, std::ostream& os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  // Global barrier state
  // std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need crts.h
  bool need_crts_h_{true};

  friend void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenSACA* p);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_SACA_H_
