#include "basic-parsers.h"
#include "expr-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "type-parsers.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

constexpr auto startfnaccLine =
    skipStuffBeforeStatement >> ("!$FNACC "_sptok || "!DIR$ FNACC "_sptok);

constexpr auto endfnaccLine = space >> endOfLine;

template <typename PA> inline constexpr auto nonemptyList(PA p) {
  return nonemptySeparated(p, ","_tok);
}

TYPE_PARSER("HOST"_tok >> pure(FnACCPackTarget::Host) ||
    "DEVICE"_tok >> pure(FnACCPackTarget::Device))

TYPE_PARSER(construct<FnACCPackClause::Item>(
    name, ":"_tok >> Parser<FnACCPackTarget>{}))

TYPE_PARSER(construct<FnACCPackClause>(
    "PACK"_tok >> parenthesized(nonemptyList(Parser<FnACCPackClause::Item>{}))))

TYPE_PARSER(construct<FnACCTileClause>(
    "TILE"_tok >> parenthesized(nonemptyList(scalarIntConstantExpr))))

TYPE_PARSER(construct<FnACCClause>(Parser<FnACCTileClause>{}) ||
    construct<FnACCClause>(Parser<FnACCPackClause>{}))

TYPE_PARSER(construct<FnACCParallelDirective>(
    "PARALLEL"_tok >> many(Parser<FnACCClause>{})))

TYPE_PARSER(construct<FnACCConstruct>(
    sourced(startfnaccLine >> Parser<FnACCParallelDirective>{} / endOfLine),
    Parser<DoConstruct>{}))

TYPE_PARSER(construct<FnACCUpdateHostDirective>(
    "UPDATE"_tok >> "HOST"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCUpdateDeviceDirective>(
    "UPDATE"_tok >> "DEVICE"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCReleaseDirective>(
    "RELEASE"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCReleaseAllDirective>(
    ("RELEASE"_tok >> "ALL"_tok >> pure(true)) ||
    ("RELEASE_ALL"_tok >> pure(true))))

TYPE_PARSER(construct<FnACCStandaloneConstruct>(startfnaccLine >>
                Parser<FnACCUpdateHostDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCUpdateDeviceDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCReleaseDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCReleaseAllDirective>{} / endOfLine))

} // namespace Fortran::parser
