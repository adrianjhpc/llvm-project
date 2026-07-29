#pragma once

#include <list>
#include <tuple>
#include <variant>

struct FnGPUPackClause {
  struct Item {
    TUPLE_CLASS_BOILERPLATE(Item);
    std::tuple<Name, FnGPUPackTarget> t;
  };
  WRAPPER_CLASS_BOILERPLATE(FnGPUPackClause, std::list<Item>);
};

struct FnGPUTileClause {
  WRAPPER_CLASS_BOILERPLATE(FnGPUTileClause, std::list<ScalarIntConstantExpr>);
};

struct FnGPUClause {
  UNION_CLASS_BOILERPLATE(FnGPUClause);
  std::variant<FnGPUTileClause, FnGPUPackClause> u; 
};

struct FnGPUParallelDirective {
  TUPLE_CLASS_BOILERPLATE(FnGPUParallelDirective);
  std::tuple<std::list<FnGPUClause>> t;
  CharBlock source;
};

struct FnGPUConstruct {
  TUPLE_CLASS_BOILERPLATE(FnGPUConstruct);
  std::tuple<FnGPUParallelDirective, DoConstruct> t; 
};


