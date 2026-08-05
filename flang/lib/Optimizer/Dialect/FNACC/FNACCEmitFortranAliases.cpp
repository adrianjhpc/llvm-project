#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCEMITFORTRANALIASES
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {

  static bool isTopLevelFlangProcedureName(StringRef name) {
    // Flang top-level procedure names are commonly emitted as _QPfoo.
    //
    // Module procedure forms such as _QMmodPfoo need different external ABI
    // aliases and are intentionally not handled by this first version.
    return name.starts_with("_QP") && name.size() > 3;
  }

  static std::string getExternalUnderscoreAlias(StringRef flangName) {
    // _QPcompute_saxpy -> compute_saxpy_
    StringRef bare = flangName.drop_front(3);
    return (bare + "_").str();
  }

  struct FNACCEmitFortranAliasesPass
    : public fir::fnacc::impl::FNACCEmitFortranAliasesBase<
    FNACCEmitFortranAliasesPass> {
    void runOnOperation() override {
      ModuleOp module = getOperation();
      MLIRContext *ctx = module.getContext();

      SymbolTable symbolTable(module);
      OpBuilder builder(ctx);

      llvm::SmallVector<func::FuncOp> functions;
      module.walk([&](func::FuncOp fn) {
	functions.push_back(fn);
      });

      for (func::FuncOp target : functions) {
	StringRef targetName = target.getSymName();

	if (!isTopLevelFlangProcedureName(targetName))
	  continue;

	// Only create aliases for definitions.
	if (target.isDeclaration())
	  continue;

	std::string aliasName = getExternalUnderscoreAlias(targetName);

	if (symbolTable.lookup(aliasName))
	  continue;

	Location loc = target.getLoc();
	FunctionType fnType = target.getFunctionType();

	builder.setInsertionPointAfter(target);

	auto alias = builder.create<func::FuncOp>(loc, aliasName, fnType);
	alias.setPublic();

	Block *entry = alias.addEntryBlock();
	builder.setInsertionPointToStart(entry);

	llvm::SmallVector<Value> args(entry->getArguments().begin(),
				      entry->getArguments().end());

	auto call = builder.create<func::CallOp>(
						 loc,
						 target.getSymName(),
						 fnType.getResults(),
						 args);

	builder.create<func::ReturnOp>(loc, call.getResults());
      }
    }
  };

} // namespace

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCEmitFortranAliasesPass() {
  return std::make_unique<FNACCEmitFortranAliasesPass>();
}

