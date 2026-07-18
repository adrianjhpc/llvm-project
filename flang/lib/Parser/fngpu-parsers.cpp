#include "parse-tree-fngpu.h"
#include "flang/Parser/parser-combinators.h"
#include "flang/Parser/token-parsers.h"

namespace Fortran::parser {

// 1. Parse the enums: "HOST" or "DEVICE"
constexpr auto fngpuPackTarget =
    "HOST" >> pure(FngpuPackTarget::Host) ||
    "DEVICE" >> pure(FngpuPackTarget::Device);

// 2. Parse a single pack item: Name : Target
TYPE_PARSER(construct<FngpuPackClause::Item>(
    name / ":", fngpuPackTarget))

// 3. Parse the whole PACK clause: PACK ( item1, item2, ... )
TYPE_PARSER(construct<FngpuPackClause>(
    "PACK" >> parenthesized(nonemptyList(Parser<FngpuPackClause::Item>{}))))

// 4. Parse the TILE clause: TILE ( int, int, ... )
TYPE_PARSER(construct<FngpuTileClause>(
    "TILE" >> parenthesized(nonemptyList(scalarIntConstantExpr))))

// 5. Parse any clause variant
TYPE_PARSER(construct<FngpuClause>(Parser<FngpuTileClause>{}) ||
            construct<FngpuClause>(Parser<FngpuPackClause>{}))

// 6. Parse the directive body: PARALLEL [clause...]
// Note: The prescanner handles the "!$FNGPU" sentinel, so this parser
// only sees the tokens that come after it.
TYPE_PARSER(construct<FngpuParallelDirective>(
    "PARALLEL" >> many(Parser<FngpuClause>{})))

// 7. Tie it together: The Directive + the Fortran Loop
TYPE_PARSER(construct<FngpuConstruct>(
    statement(Parser<FngpuParallelDirective>{}),
    indirect(Parser<ExecutableConstruct>{})))

} // namespace Fortran::parser
