#pragma once

#include <list>
#include <tuple>
#include <variant>

ENUM_CLASS(FnACCPackTarget, Host, Device);

struct FnACCPackClause {
  struct Item {
    TUPLE_CLASS_BOILERPLATE(Item);
    std::tuple<Name, FnACCPackTarget> t;
  };
  WRAPPER_CLASS_BOILERPLATE(FnACCPackClause, std::list<Item>);
};

struct FnACCTileClause {
  WRAPPER_CLASS_BOILERPLATE(FnACCTileClause, std::list<ScalarIntConstantExpr>);
};

struct FnACCClause {
  UNION_CLASS_BOILERPLATE(FnACCClause);
  std::variant<FnACCTileClause, FnACCPackClause> u;
};

struct FnACCParallelDirective {
  TUPLE_CLASS_BOILERPLATE(FnACCParallelDirective);
  std::tuple<std::list<FnACCClause>> t;
  CharBlock source;
};

struct FnACCConstruct {
  TUPLE_CLASS_BOILERPLATE(FnACCConstruct);
  std::tuple<FnACCParallelDirective, DoConstruct> t;
};

// FNACC standalone data-management directives.
//
// Source forms:
//
//   !$fnacc update host(a, b)
//   !$fnacc update device(a, b)
//   !$fnacc release(a, b)
//   !$fnacc release all

struct FnACCUpdateHostDirective {
  TUPLE_CLASS_BOILERPLATE(FnACCUpdateHostDirective);
  std::tuple<std::list<Name>> t;
};

struct FnACCUpdateDeviceDirective {
  TUPLE_CLASS_BOILERPLATE(FnACCUpdateDeviceDirective);
  std::tuple<std::list<Name>> t;
};

struct FnACCReleaseDirective {
  TUPLE_CLASS_BOILERPLATE(FnACCReleaseDirective);
  std::tuple<std::list<Name>> t;
};

struct FnACCReleaseAllDirective {
  WRAPPER_CLASS_BOILERPLATE(FnACCReleaseAllDirective, bool);
};

struct FnACCStandaloneConstruct {
  UNION_CLASS_BOILERPLATE(FnACCStandaloneConstruct);
  std::variant<FnACCUpdateHostDirective, FnACCUpdateDeviceDirective,
      FnACCReleaseDirective, FnACCReleaseAllDirective>
      u;
};
