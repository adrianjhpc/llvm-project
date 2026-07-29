#include "flang/Parser/parse-tree.h"
#include "basic-parsers.h"
#include "expr-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parsers.h"
#include "type-parser-implementation.h"

namespace Fortran::parser {

constexpr auto startFngpuLine = skipStuffBeforeStatement >> ("!$FNGPU "_sptok || "!DIR$ FNGPU "_sptok);
constexpr auto endFngpuLine = space >> endOfLine;

template <typename PA> inline constexpr auto nonemptyList(PA p) {
  return nonemptySeparated(p, ","_tok);
}

TYPE_PARSER(
    "HOST"_tok >> pure(FnGPUPackTarget::Host) ||
    "DEVICE"_tok >> pure(FnGPUPackTarget::Device)
)

TYPE_PARSER(construct<FnGPUPackClause::Item>(
    name, ":"_tok >> Parser<FnGPUPackTarget>{}
))

TYPE_PARSER(construct<FnGPUPackClause>(
    "PACK"_tok >> parenthesized(nonemptyList(Parser<FnGPUPackClause::Item>{}))
))

TYPE_PARSER(construct<FnGPUTileClause>(
    "TILE"_tok >> parenthesized(nonemptyList(scalarIntConstantExpr))
))

TYPE_PARSER(
    construct<FnGPUClause>(Parser<FnGPUTileClause>{}) ||
    construct<FnGPUClause>(Parser<FnGPUPackClause>{})
)

TYPE_PARSER(construct<FnGPUParallelDirective>(
    "PARALLEL"_tok >> many(Parser<FnGPUClause>{})
))

TYPE_PARSER(construct<FnGPUConstruct>(
    sourced(startFngpuLine >> Parser<FnGPUParallelDirective>{} / endOfLine),
    Parser<DoConstruct>{}
))

} // namespace Fortran::parser
