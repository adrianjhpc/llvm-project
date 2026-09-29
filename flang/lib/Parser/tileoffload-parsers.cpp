#include "basic-parsers.h"
#include "expr-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "type-parsers.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

constexpr auto startTileOffloadLine =
    skipStuffBeforeStatement >> ("!$tileoff "_sptok || "!@tileoff "_sptok);

constexpr auto endTileOffloadLine = space >> endOfLine;

template <typename PA> inline constexpr auto nonemptyList(PA p) {
  return nonemptySeparated(p, ","_tok);
}

TYPE_PARSER("HOST"_tok >> pure(TileOffloadPackTarget::Host) ||
    "DEVICE"_tok >> pure(TileOffloadPackTarget::Device))

TYPE_PARSER(construct<TileOffloadPackClause::Item>(
    name, ":"_tok >> Parser<TileOffloadPackTarget>{}))

TYPE_PARSER(construct<TileOffloadPackClause>(
    "PACK"_tok >> parenthesized(nonemptyList(Parser<TileOffloadPackClause::Item>{}))))

TYPE_PARSER("+"_tok >> pure(TileOffloadReductionOperator::Add) ||
    "*"_tok >> pure(TileOffloadReductionOperator::Multiply) ||
    "MIN"_tok >> pure(TileOffloadReductionOperator::Min) ||
    "MAX"_tok >> pure(TileOffloadReductionOperator::Max))

TYPE_PARSER(construct<TileOffloadReductionClause::Item>(
    Parser<TileOffloadReductionOperator>{}, ":"_tok >> name))

TYPE_PARSER(construct<TileOffloadReductionClause>("REDUCTION"_tok >>
    parenthesized(nonemptyList(Parser<TileOffloadReductionClause::Item>{}))))

TYPE_PARSER(construct<TileOffloadTileClause>(
    "TILE"_tok >> parenthesized(nonemptyList(scalarIntConstantExpr))))

TYPE_PARSER(construct<TileOffloadNoCopybackClause>("NO_COPYBACK"_tok >> pure(true)))

TYPE_PARSER("IEEE"_tok >> pure(TileOffloadMatmulPrecision::IEEE) ||
    "TF32X3"_tok >> pure(TileOffloadMatmulPrecision::TF32x3) ||
    "TF32"_tok >> pure(TileOffloadMatmulPrecision::TF32))

TYPE_PARSER(construct<TileOffloadMatmulPrecisionClause>(
    "MATMUL_PRECISION"_tok >> parenthesized(Parser<TileOffloadMatmulPrecision>{})))

TYPE_PARSER(construct<TileOffloadClause>(Parser<TileOffloadTileClause>{}) ||
    construct<TileOffloadClause>(Parser<TileOffloadPackClause>{}) ||
    construct<TileOffloadClause>(Parser<TileOffloadReductionClause>{}) ||
    construct<TileOffloadClause>(Parser<TileOffloadNoCopybackClause>{}) ||
    construct<TileOffloadClause>(Parser<TileOffloadMatmulPrecisionClause>{}))

TYPE_PARSER(construct<TileOffloadParallelDirective>(
    "PARALLEL"_tok >> many(Parser<TileOffloadClause>{})))

TYPE_PARSER(construct<TileOffloadConstruct>(
    sourced(startTileOffloadLine >> Parser<TileOffloadParallelDirective>{} / endOfLine),
    Parser<DoConstruct>{}))

TYPE_PARSER(construct<TileOffloadUpdateHostDirective>(
    "UPDATE"_tok >> "HOST"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadUpdateDeviceDirective>(
    "UPDATE"_tok >> "DEVICE"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadPresentDirective>(
    "PRESENT"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadReleaseDirective>(
    "RELEASE"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadReleaseAllDirective>(
    ("RELEASE"_tok >> "ALL"_tok >> pure(true)) ||
    ("RELEASE_ALL"_tok >> pure(true))))

TYPE_PARSER(construct<TileOffloadWaitDirective>("WAIT"_tok >> pure(true)))

TYPE_PARSER(construct<TileOffloadCopyinClause>(
    "COPYIN"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadCreateClause>(
    "CREATE"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadCopyoutClause>(
    "COPYOUT"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadDeleteClause>(
    "DELETE"_tok >> parenthesized(nonemptyList(variable))))

TYPE_PARSER(construct<TileOffloadEnterDataClause>(Parser<TileOffloadCopyinClause>{}) ||
    construct<TileOffloadEnterDataClause>(Parser<TileOffloadCreateClause>{}))

TYPE_PARSER(construct<TileOffloadExitDataClause>(Parser<TileOffloadCopyoutClause>{}) ||
    construct<TileOffloadExitDataClause>(Parser<TileOffloadDeleteClause>{}))

TYPE_PARSER(
    sourced(construct<TileOffloadEnterDataDirective>("ENTER"_tok >> "DATA"_tok >>
        many(construct<TileOffloadEnterDataClause>(Parser<TileOffloadCopyinClause>{}) ||
            construct<TileOffloadEnterDataClause>(Parser<TileOffloadCreateClause>{})))))

TYPE_PARSER(
    sourced(construct<TileOffloadExitDataDirective>("EXIT"_tok >> "DATA"_tok >>
        many(construct<TileOffloadExitDataClause>(Parser<TileOffloadCopyoutClause>{}) ||
            construct<TileOffloadExitDataClause>(Parser<TileOffloadDeleteClause>{})))))

TYPE_PARSER(construct<TileOffloadStandaloneConstruct>(startTileOffloadLine >>
                Parser<TileOffloadEnterDataDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadExitDataDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadUpdateHostDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadUpdateDeviceDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadPresentDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadReleaseAllDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadReleaseDirective>{} / endOfLine) ||
    construct<TileOffloadStandaloneConstruct>(
        startTileOffloadLine >> Parser<TileOffloadWaitDirective>{} / endOfLine))

} // namespace Fortran::parser
