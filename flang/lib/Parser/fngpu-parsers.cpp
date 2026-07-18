#include "parse-tree-fngpu.h"
#include "flang/Parser/parser-combinators.h"
#include "flang/Parser/token-parsers.h"

namespace Fortran::parser {

// 1. Parse the enums: "HOST" or "DEVICE"
constexpr auto fngpuPackTarget =
    "HOST" >> pure(FnGPUPackTarget::Host) ||
    "DEVICE" >> pure(FnGPUPackTarget::Device);

// 2. Parse a single pack item: Name : Target
TYPE_PARSER(construct<FnGPUPackClause::Item>(
    name / ":", fngpuPackTarget))

// 3. Parse the whole PACK clause: PACK ( item1, item2, ... )
TYPE_PARSER(construct<FnGPUPackClause>(
    "PACK" >> parenthesized(nonemptyList(Parser<FnGPUPackClause::Item>{}))))

// 4. Parse the TILE clause: TILE ( int, int, ... )
TYPE_PARSER(construct<FnGPUTileClause>(
    "TILE" >> parenthesized(nonemptyList(scalarIntConstantExpr))))

// 5. Parse any clause variant
TYPE_PARSER(construct<FnGPUClause>(Parser<FnGPUTileClause>{}) ||
            construct<FnGPUClause>(Parser<FnGPUPackClause>{}))

// 6. Parse the directive body: PARALLEL [clause...]
// Note: The prescanner handles the "!$FNGPU" sentinel, so this parser
// only sees the tokens that come after it.
TYPE_PARSER(construct<FnGPUParallelDirective>(
    "PARALLEL" >> many(Parser<FnGPUClause>{})))

// 7. Tie it together: The Directive + the Fortran Loop
TYPE_PARSER(construct<FnGPUConstruct>(
    statement(Parser<FnGPUParallelDirective>{}),
    indirect(Parser<ExecutableConstruct>{})))

} // namespace Fortran::parser
