/*
    Level Schedule
*/

#include "../utils.h"

namespace tvm {
namespace tir {

class HasAnnotationOrThreadBindingError : public ScheduleError {
 public:
  explicit HasAnnotationOrThreadBindingError(IRModule mod, For loop)
      : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive can't be applied because the loop has annotation or "
           "thread binding";
  }

  String DetailRenderTemplate() const final {
    return "The primitive can't be applied because the loop {0} has annotation or thread binding";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class LoopNotStartWithZeroError : public ScheduleError {
 public:
  explicit LoopNotStartWithZeroError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The primitive only supports loop starting with 0";
  }

  String DetailRenderTemplate() const final {
    return "The loop {0} does not start with 0, which is not supported";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

/*! \brief Substitute vars and collect the reuse mapping of opaque blocks */
class SubstituteVarAndCollectOpaqueBlock : public StmtExprMutator {
 public:
  explicit SubstituteVarAndCollectOpaqueBlock(std::function<Optional<PrimExpr>(const Var&)> vmap,
                                              Map<Block, Block>* opaque_blocks)
      : vmap_(vmap), opaque_blocks_(opaque_blocks) {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (Optional<PrimExpr> ret = vmap_(var)) {
      return ret.value();
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    if (realize->block->iter_vars.empty()) {
      opaque_blocks_->Set(op->block, realize->block);
    }
    return std::move(realize);
  }

  /*! \brief The substitute function */
  std::function<Optional<PrimExpr>(const Var&)> vmap_;
  /*! \brief The reuse mapping of opaque blocks */
  Map<Block, Block>* opaque_blocks_;
};

/*! \brief Append a new predicate to the each child of type BlockRealize (not recursively) */
class BlockPredicateAppender : public StmtMutator {
 public:
  /*!
   * \brief Constructor
   * \param to_append The predicate to be appended to BlockRealizeNode
   */
  explicit BlockPredicateAppender(const PrimExpr& to_append) : to_append_(to_append) {}

 private:
  // For each direct child of type BlockRealizeNode, append the predicate
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // We do not recursively do this
    ObjectPtr<BlockRealizeNode> n = CopyOnWrite(realize);
    n->predicate = n->predicate && to_append_;
    return BlockRealize(n);
  }

  /*! \brief The predicate to be appended */
  const PrimExpr& to_append_;
};

/*! \brief Simplify the binding of block realize and update the opaque block reuse mapping */
class IterMapSimplifyBlockBindingLS : public StmtExprMutator {
 public:
  explicit IterMapSimplifyBlockBindingLS(MapNode* opaque_blocks, Map<Var, Range> loop_var2extent,
                                         const PrimExpr& idx)
      : opaque_blocks_(opaque_blocks), loop_var2extent_(loop_var2extent), idx_(idx) {}

  static For SimplifyBindings(Stmt stmt, const Array<StmtSRef>& loop_srefs, MapNode* opaque_blocks,
                              const PrimExpr& idx) {
    if (loop_srefs.size() != 0)
      LOG(FATAL)
          << "Level schedule only support the dependent traversal of the outermost single for loop";
    Map<Var, Range> loop_var2extent;
    return Downcast<For>(IterMapSimplifyBlockBindingLS(opaque_blocks, std::move(loop_var2extent),
                                                       idx)(std::move(stmt)));
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_var2extent_.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    Stmt res = StmtMutator::VisitStmt_(op);
    loop_var2extent_.erase(op->loop_var);
    return res;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // skip opaque block and update mapping
    if (op->iter_values.empty()) {
      Block block = op->block;
      BlockRealize realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
      for (const std::pair<ObjectRef, ObjectRef>& entry : *opaque_blocks_) {
        if (entry.second.same_as(block)) {
          opaque_blocks_->at(entry.first) = realize->block;
          break;
        }
      }
      return std::move(realize);
    }

    if (op->iter_values.size() != 1 || !idx_.same_as(op->iter_values[0])) {
      LOG(FATAL) << "Level schedule only support single iter values and must equal to outer "
                    "binding iter var";
    }

    Array<PrimExpr> v = arith::IterMapSimplify(/*indices= */ op->iter_values,
                                               /*input_iters=*/loop_var2extent_,
                                               /*input_pred=*/op->predicate,
                                               /*require_bijective=*/false);

    if (v.same_as(op->iter_values)) {
      return GetRef<Stmt>(op);
    } else {
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->iter_values = std::move(v);
      return Stmt(n);
    }
  }

  /*! \brief The reuse mapping */
  MapNode* opaque_blocks_;
  /*! \brief The range of loops */
  Map<Var, Range> loop_var2extent_;
  /* substitute index */
  PrimExpr idx_;
};

/******** Implementation ********/

void LevelSchedule(ScheduleState self, const StmtSRef& loop_sref, int level_number,
                   const Buffer& level_num_buf, const Buffer& level_idx_buf) {
  // Step 1. Check correctness
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (!loop->annotations.empty() || loop->thread_binding.defined()) {
    throw HasAnnotationOrThreadBindingError(self->mod, GetRef<For>(loop));
  }
  // Currently, loops not starting with 0 are not supported
  arith::Analyzer analyzer;
  if (!analyzer.CanProve(loop->min == 0)) {
    throw LoopNotStartWithZeroError(self->mod, GetRef<For>(loop));
  }
  // Step 2. Replace with two loops
  int n = 2;
  bool isreversed = false;
  PrimExpr substitute_value;
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);

  if (level_number < 0) {
    isreversed = true;
    level_number = -level_number;
  }
  // outer level range
  const PrimExpr& level = level_number;
  Var var = loop->loop_var.copy_with_suffix("_level");
  substitute_value = isreversed ? level_number - 1 - var : var;
  analyzer.Bind(var, Range::FromMinExtent(0, level));
  new_loop_vars.emplace_back(std::move(var));

  // level_num range
  Array<PrimExpr> level_num_indices, level_num_indices_add_one;
  level_num_indices.push_back(substitute_value);
  level_num_indices_add_one.push_back(substitute_value + 1);
  const PrimExpr& low_bound = level_num_buf.vload(level_num_indices, level_num_buf->dtype);
  const PrimExpr& level_num_extent =
      level_num_buf.vload(level_num_indices_add_one, level_num_buf->dtype) - low_bound;
  Var level_num_var = loop->loop_var.copy_with_suffix("_level_num");

  Array<PrimExpr> level_idx_indices;
  level_idx_indices.push_back(low_bound + level_num_var);
  const PrimExpr& cur_idx = level_idx_buf.vload(level_idx_indices, level_idx_buf->dtype);
  substitute_value = cur_idx;
  analyzer.Bind(level_num_var, Range::FromMinExtent(0, level_num_extent));
  new_loop_vars.emplace_back(std::move(level_num_var));

  Map<Block, Block> opaque_block_reuse;
  Stmt new_stmt = loop->body;
  new_stmt = SubstituteVarAndCollectOpaqueBlock(
      [&](const Var& v) -> Optional<PrimExpr> {
        if (v.same_as(loop->loop_var)) {
          return substitute_value;
        } else {
          return NullOpt;
        }
      },
      &opaque_block_reuse)(std::move(new_stmt));
  // Step 3. Update predicate to guard the loop
  PrimExpr predicate = substitute_value < loop->extent;
  if (!analyzer.CanProve(predicate)) {
    new_stmt = BlockPredicateAppender(/*predicate=*/predicate)(std::move(new_stmt));
  }
  // Step 4. Generate nested loops to replace the original loop and simplify the binding
  new_stmt = For(new_loop_vars[1], 0, level_num_extent, ForKind::kSerial, new_stmt);
  new_stmt = For(new_loop_vars[0], 0, level, ForKind::kSerial, new_stmt);
  new_stmt = IterMapSimplifyBlockBindingLS::SimplifyBindings(
      std::move(new_stmt), GetLoops(loop_sref), opaque_block_reuse.CopyOnWrite(), substitute_value);
  self->Replace(loop_sref, new_stmt, opaque_block_reuse);
}

/******** Instruction Registration ********/

struct LevelScheduleTraits : public UnpackedInstTraits<LevelScheduleTraits> {
  static constexpr const char* kName = "LevelSchedule";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop, Integer level_number,
                                      Buffer level_num_buf, Buffer level_idx_buf) {
    return sch->LevelSchedule(loop, level_number, level_num_buf, level_idx_buf);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop, Integer level_number,
                                 String level_num_buf, String level_idx_buf) {
    PythonAPICall py("level_schedule");
    py.Input("loop", loop);
    py.Input("level_number", level_number);
    py.Input("level_num_buf", level_num_buf);
    py.Input("level_idx_buf", level_idx_buf);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(LevelScheduleTraits);
}  // namespace tir
}  // namespace tvm