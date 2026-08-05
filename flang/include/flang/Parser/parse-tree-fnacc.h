#pragma once

#include <list>
#include <tuple>
#include <variant>

ENUM_CLASS(FnGPUPackTarget, Host, Device);

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

// FNGPU standalone data-management directives.
//
// Source forms:
//
//   !$fngpu update host(a, b)
//   !$fngpu update device(a, b)
//   !$fngpu release(a, b)
//   !$fngpu release all

struct FnGPUUpdateHostDirective {
  TUPLE_CLASS_BOILERPLATE(FnGPUUpdateHostDirective);
  std::tuple<std::list<Name>> t;
};

struct FnGPUUpdateDeviceDirective {
  TUPLE_CLASS_BOILERPLATE(FnGPUUpdateDeviceDirective);
  std::tuple<std::list<Name>> t;
};

struct FnGPUReleaseDirective {
  TUPLE_CLASS_BOILERPLATE(FnGPUReleaseDirective);
  std::tuple<std::list<Name>> t;
};

struct FnGPUReleaseAllDirective {
  WRAPPER_CLASS_BOILERPLATE(FnGPUReleaseAllDirective, bool);
};

struct FnGPUStandaloneConstruct {
  UNION_CLASS_BOILERPLATE(FnGPUStandaloneConstruct);
  std::variant<FnGPUUpdateHostDirective, FnGPUUpdateDeviceDirective,
      FnGPUReleaseDirective, FnGPUReleaseAllDirective>
      u;
};
