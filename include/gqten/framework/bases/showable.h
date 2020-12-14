// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-14 09:46
*
* Description: GraceQ/tensor project. Abstract base class for object which need
* be format print.
*/

/**
@file showable.h
@brief  Abstract base class for object which need be format print.
*/
#ifndef GQTEN_FRAMEWORK_BASES_SHOWABLE_H
#define GQTEN_FRAMEWORK_BASES_SHOWABLE_H


#include <string>       // string
#include <iostream>     // ostream


namespace gqten {


/// Indent character.
const std::string kIndentCharacter = "  ";


/**
Abstract base class for object which need be format print.
*/
class Showable {
public:
  Showable(void) = default;
  virtual ~Showable(void) = default;

  /// Show the object to standard output.
  virtual void Show(const size_t) const = 0;
};


/**
Format print a object.

@param obj A showable object.
@param indent_level Indentation level.
*/
inline void Show(const Showable &obj, const size_t indent_level = 0) {
  obj.Show(indent_level);
}


class IndentPrinter {
public:
  IndentPrinter(const size_t indent_level) : indent_level_(indent_level) {}

  void PrintIndent(std::ostream &os) const {
    for (size_t i = 0; i < indent_level_; ++i) {
      os << kIndentCharacter;
    }
  }

private:
  size_t indent_level_;
};


inline std::ostream &operator<<(
    std::ostream &os,
    const IndentPrinter &indent_printer
) {
  indent_printer.PrintIndent(os);
  return os;
}
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_BASES_SHOWABLE_H */
