#include "basic-parsers.h"
#include "expr-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "type-parsers.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

constexpr auto startfnaccLine = skipStuffBeforeStatement >>
    ("!$FNACC "_sptok || "!@FNACC "_sptok || "!DIR$ FNACC "_sptok);

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

TYPE_PARSER("+"_tok >> pure(FnACCReductionOperator::Add))

TYPE_PARSER(construct<FnACCReductionClause::Item>(
    Parser<FnACCReductionOperator>{}, ":"_tok >> name))

TYPE_PARSER(construct<FnACCReductionClause>("REDUCTION"_tok >>
    parenthesized(nonemptyList(Parser<FnACCReductionClause::Item>{}))))

TYPE_PARSER(construct<FnACCTileClause>(
    "TILE"_tok >> parenthesized(nonemptyList(scalarIntConstantExpr))))

TYPE_PARSER(construct<FnACCClause>(Parser<FnACCTileClause>{}) ||
    construct<FnACCClause>(Parser<FnACCPackClause>{}) ||
    construct<FnACCClause>(Parser<FnACCReductionClause>{}))

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

TYPE_PARSER(construct<FnACCCopyinClause>(
    "COPYIN"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCCreateClause>(
    "CREATE"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCCopyoutClause>(
    "COPYOUT"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCDeleteClause>(
    "DELETE"_tok >> parenthesized(nonemptyList(name))))

TYPE_PARSER(construct<FnACCEnterDataClause>(Parser<FnACCCopyinClause>{}) ||
    construct<FnACCEnterDataClause>(Parser<FnACCCreateClause>{}))

TYPE_PARSER(construct<FnACCExitDataClause>(Parser<FnACCCopyoutClause>{}) ||
    construct<FnACCExitDataClause>(Parser<FnACCDeleteClause>{}))

TYPE_PARSER(construct<FnACCEnterDataDirective>(
    "ENTER"_tok >> "DATA"_tok >> many(Parser<FnACCEnterDataClause>{})))

TYPE_PARSER(construct<FnACCExitDataDirective>(
    "EXIT"_tok >> "DATA"_tok >> many(Parser<FnACCExitDataClause>{})))

TYPE_PARSER(construct<FnACCStandaloneConstruct>(startfnaccLine >>
                Parser<FnACCEnterDataDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCExitDataDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCUpdateHostDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCUpdateDeviceDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCReleaseDirective>{} / endOfLine) ||
    construct<FnACCStandaloneConstruct>(
        startfnaccLine >> Parser<FnACCReleaseAllDirective>{} / endOfLine))

} // namespace Fortran::parser
