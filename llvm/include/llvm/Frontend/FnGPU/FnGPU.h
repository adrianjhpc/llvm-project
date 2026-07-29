#pragma once

#include "flang/Parser/parse-tree.h"
#include <list>
#include <tuple>
#include <variant>

namespace Fortran::parser {

// Enum to represent the packing destination
ENUM_CLASS(FnGPUPackTarget, Host, Device)

} // namespace Fortran::parser
