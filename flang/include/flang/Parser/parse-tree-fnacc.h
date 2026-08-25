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

ENUM_CLASS(FnACCReductionOperator, Add, Multiply, Min, Max);

struct FnACCReductionClause {
  struct Item {
    TUPLE_CLASS_BOILERPLATE(Item);
    std::tuple<FnACCReductionOperator, Name> t;
  };

  WRAPPER_CLASS_BOILERPLATE(FnACCReductionClause, std::list<Item>);
};

struct FnACCClause {
  UNION_CLASS_BOILERPLATE(FnACCClause);
  std::variant<FnACCTileClause, FnACCPackClause, FnACCReductionClause> u;
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
//   !$fnacc wait

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

struct FnACCWaitDirective {
  WRAPPER_CLASS_BOILERPLATE(FnACCWaitDirective, bool);
};

struct FnACCCopyinClause {
  WRAPPER_CLASS_BOILERPLATE(FnACCCopyinClause, std::list<Name>);
};

struct FnACCCreateClause {
  WRAPPER_CLASS_BOILERPLATE(FnACCCreateClause, std::list<Name>);
};

struct FnACCCopyoutClause {
  WRAPPER_CLASS_BOILERPLATE(FnACCCopyoutClause, std::list<Name>);
};

struct FnACCDeleteClause {
  WRAPPER_CLASS_BOILERPLATE(FnACCDeleteClause, std::list<Name>);
};

struct FnACCEnterDataClause {
  UNION_CLASS_BOILERPLATE(FnACCEnterDataClause);
  std::variant<FnACCCopyinClause, FnACCCreateClause> u;
};

struct FnACCExitDataClause {
  UNION_CLASS_BOILERPLATE(FnACCExitDataClause);
  std::variant<FnACCCopyoutClause, FnACCDeleteClause> u;
};

struct FnACCEnterDataDirective {
  TUPLE_CLASS_BOILERPLATE(FnACCEnterDataDirective);
  std::tuple<std::list<FnACCEnterDataClause>> t;
  CharBlock source;
};

struct FnACCExitDataDirective {
  TUPLE_CLASS_BOILERPLATE(FnACCExitDataDirective);
  std::tuple<std::list<FnACCExitDataClause>> t;
  CharBlock source;
};

struct FnACCStandaloneConstruct {
  UNION_CLASS_BOILERPLATE(FnACCStandaloneConstruct);
  std::variant<FnACCUpdateHostDirective, FnACCUpdateDeviceDirective,
      FnACCReleaseDirective, FnACCReleaseAllDirective, FnACCEnterDataDirective,
      FnACCExitDataDirective, FnACCWaitDirective>
      u;
  CharBlock source;
};
