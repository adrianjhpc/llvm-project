#pragma once

#include "flang/Parser/parse-tree.h"
#include <list>
#include <tuple>
#include <variant>

namespace Fortran::parser {

// Enum to represent the packing destination
ENUM_CLASS(FngpuPackTarget, Host, Device)

// Represents: PACK(A: HOST, B: DEVICE)
struct FngpuPackClause {
  TUPLE_CLASS_BOILERPLATE(FngpuPackClause);
  struct Item {
    TUPLE_CLASS_BOILERPLATE(Item);
    std::tuple<Name, FngpuPackTarget> t; // Pair of array name and target
  };
  std::list<Item> v;
};

// Represents: TILE(128, 64)
struct FngpuTileClause {
  TUPLE_CLASS_BOILERPLATE(FngpuTileClause);
  std::list<ScalarIntConstantExpr> v; // List of integer tile dimensions
};

// A variant holding any valid FNGPU clause
struct FngpuClause {
  UNION_CLASS_BOILERPLATE(FngpuClause);
  std::variant<FngpuTileClause, FngrpPackClause> u;
};

// Represents the standalone directive: !$FNGPU PARALLEL [clauses]
struct FngpuParallelDirective {
  TUPLE_CLASS_BOILERPLATE(FngpuParallelDirective);
  std::tuple<std::list<FngpuClause>> t;
};

// Represents the directive AND the Fortran loop block it applies to
struct FngpuConstruct {
  TUPLE_CLASS_BOILERPLATE(FngpuConstruct);
  std::tuple<Statement<FngpuParallelDirective>,
             common::Indirection<ExecutableConstruct>> t; 
};

} // namespace Fortran::parser
