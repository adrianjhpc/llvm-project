include "llvm/Frontend/Directive/DirectiveBase.td"

def FnGPUDirectiveLanguage : DirectiveLanguage {
  let name = "FnGPU";
  let cppNamespace = "fngpu";
  let directivePrefix = "FNGPU_";
  let clausePrefix = "FNGPU_";
  let includeHeader = "llvm/Frontend/FnGPU/FnGPU.h.inc";
  let flangClauseBaseClass = "FnGPUClause";
}
