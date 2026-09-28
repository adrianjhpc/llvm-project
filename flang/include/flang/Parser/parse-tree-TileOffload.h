#pragma once

#include <list>
#include <tuple>
#include <variant>

ENUM_CLASS(TileOffloadPackTarget, Host, Device);

struct TileOffloadPackClause {
  struct Item {
    TUPLE_CLASS_BOILERPLATE(Item);
    std::tuple<Name, TileOffloadPackTarget> t;
  };
  WRAPPER_CLASS_BOILERPLATE(TileOffloadPackClause, std::list<Item>);
};

struct TileOffloadTileClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadTileClause, std::list<ScalarIntConstantExpr>);
};

struct TileOffloadNoCopybackClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadNoCopybackClause, bool);
};

ENUM_CLASS(TileOffloadMatmulPrecision, IEEE, TF32, TF32x3);

struct TileOffloadMatmulPrecisionClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadMatmulPrecisionClause, TileOffloadMatmulPrecision);
};

ENUM_CLASS(TileOffloadReductionOperator, Add, Multiply, Min, Max);

struct TileOffloadReductionClause {
  struct Item {
    TUPLE_CLASS_BOILERPLATE(Item);
    std::tuple<TileOffloadReductionOperator, Name> t;
  };

  WRAPPER_CLASS_BOILERPLATE(TileOffloadReductionClause, std::list<Item>);
};

struct TileOffloadClause {
  UNION_CLASS_BOILERPLATE(TileOffloadClause);
  std::variant<TileOffloadTileClause, TileOffloadPackClause, TileOffloadReductionClause,
      TileOffloadNoCopybackClause, TileOffloadMatmulPrecisionClause>
      u;
};

struct TileOffloadParallelDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadParallelDirective);
  std::tuple<std::list<TileOffloadClause>> t;
  CharBlock source;
};

struct TileOffloadConstruct {
  TUPLE_CLASS_BOILERPLATE(TileOffloadConstruct);
  std::tuple<TileOffloadParallelDirective, DoConstruct> t;
};

// TileOffload standalone data-management directives.
//
// Source forms:
//
//   !$tileoff update host(a, b)
//   !$tileoff update device(a, b)
//   !$tileoff release(a, b)
//   !$tileoff release all
//   !$tileoff wait

struct TileOffloadUpdateHostDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadUpdateHostDirective);
  std::tuple<std::list<Variable>> t;
};

struct TileOffloadUpdateDeviceDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadUpdateDeviceDirective);
  std::tuple<std::list<Variable>> t;
};

/// Assert that each named object already has a live TileOffload device allocation.
/// This directive never allocates or transfers data.
struct TileOffloadPresentDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadPresentDirective);
  std::tuple<std::list<Variable>> t;
};

struct TileOffloadReleaseDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadReleaseDirective);
  std::tuple<std::list<Variable>> t;
};

struct TileOffloadReleaseAllDirective {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadReleaseAllDirective, bool);
};

struct TileOffloadWaitDirective {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadWaitDirective, bool);
};

struct TileOffloadCopyinClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadCopyinClause, std::list<Variable>);
};

struct TileOffloadCreateClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadCreateClause, std::list<Variable>);
};

struct TileOffloadCopyoutClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadCopyoutClause, std::list<Variable>);
};

struct TileOffloadDeleteClause {
  WRAPPER_CLASS_BOILERPLATE(TileOffloadDeleteClause, std::list<Variable>);
};

struct TileOffloadEnterDataClause {
  UNION_CLASS_BOILERPLATE(TileOffloadEnterDataClause);
  std::variant<TileOffloadCopyinClause, TileOffloadCreateClause> u;
};

struct TileOffloadExitDataClause {
  UNION_CLASS_BOILERPLATE(TileOffloadExitDataClause);
  std::variant<TileOffloadCopyoutClause, TileOffloadDeleteClause> u;
};

struct TileOffloadEnterDataDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadEnterDataDirective);
  std::tuple<std::list<TileOffloadEnterDataClause>> t;
  CharBlock source;
};

struct TileOffloadExitDataDirective {
  TUPLE_CLASS_BOILERPLATE(TileOffloadExitDataDirective);
  std::tuple<std::list<TileOffloadExitDataClause>> t;
  CharBlock source;
};

struct TileOffloadStandaloneConstruct {
  UNION_CLASS_BOILERPLATE(TileOffloadStandaloneConstruct);
  std::variant<TileOffloadUpdateHostDirective, TileOffloadUpdateDeviceDirective,
      TileOffloadPresentDirective, TileOffloadReleaseDirective, TileOffloadReleaseAllDirective,
      TileOffloadEnterDataDirective, TileOffloadExitDataDirective, TileOffloadWaitDirective>
      u;
  CharBlock source;
};
