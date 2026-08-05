#include "basic-parsers.h"
#include "expr-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "type-parsers.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

constexpr auto startFngpuLine =
    skipStuffBeforeStatement >> ("!$FNGPU "_sptok || "!DIR$ FNGPU "_sptok);

constexpr auto endFngpuLine = space >> endOfLine;

template <typename PA> inline constexpr auto nonemptyList(PA p) {
  return nonemptySeparated(p, ","_tok);
}

TYPE_PARSER("HOST"_tok >> pure(FnGPUPackTarget::Host) ||
    "DEVICE"_tok >> pure(FnGPUPackTarget::Device))

TYPE_PARSER(construct<FnGPUPackClause::Item>(
    name, ":"_tok >> Parser<FnGPUPackTarget>{}))

TYPE_PARSER(construct<FnGPUPackClause>(
    "PACK"_tok >> parenthesized(nonemptyList(Parser<FnGPUPackClause::Item>{}))))

TYPE_PARSER(construct<FnGPUTileClause>(
    "TILE"_tok >> parenthesized(nonemptyList(scalarIntConstantExpr))))

TYPE_PARSER(construct<FnGPUClause>(Parser<FnGPUTileClause>{}) ||
    construct<FnGPUClause>(Parser<FnGPUPackClause>{}))

TYPE_PARSER(construct<FnGPUParallelDirective>(
    "PARALLEL"_tok >> many(Parser<FnGPUClause>{})))

TYPE_PARSER(construct<FnGPUConstruct>(
    sourced(startFngpuLine >> Parser<FnGPUParallelDirective>{} / endOfLine),
    Parser<DoConstruct>{}))

TYPE_PARSER(construct<FnGPUUpdateHostDirective>(
    "UPDATE"_tok >> "HOST"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnGPUUpdateDeviceDirective>(
    "UPDATE"_tok >> "DEVICE"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnGPUReleaseDirective>(
    "RELEASE"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnGPUReleaseAllDirective>(
    ("RELEASE"_tok >> "ALL"_tok >> pure(true)) ||
    ("RELEASE_ALL"_tok >> pure(true))))

TYPE_PARSER(construct<FnGPUStandaloneConstruct>(startFngpuLine >>
                Parser<FnGPUUpdateHostDirective>{} / endOfLine) ||
    construct<FnGPUStandaloneConstruct>(
        startFngpuLine >> Parser<FnGPUUpdateDeviceDirective>{} / endOfLine) ||
    construct<FnGPUStandaloneConstruct>(
        startFngpuLine >> Parser<FnGPUReleaseDirective>{} / endOfLine) ||
    construct<FnGPUStandaloneConstruct>(
        startFngpuLine >> Parser<FnGPUReleaseAllDirective>{} / endOfLine))

} // namespace Fortran::parser
