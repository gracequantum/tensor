// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-30 10:31
*
* Description: GraceQ/tensor project. Quantum number value with U(1) symmetry.
*/

/**
@file quval_u1.h
@brief Quantum number value with U1 symmetry.
*/
#ifndef GQTEN_GQTENSOR_QN_QNVAL_U1_H
#define GQTEN_GQTENSOR_QN_QNVAL_U1_H


#include "gqten/gqtensor/qn/qnval.h"    // QNVal


namespace gqten {


/// Dimension of the U(1) group representation.
const size_t kDimOfU1Repr = 1;


/**
Quantum number value which represents a U(1) group representation.

@param val The label of the U(1) representation.
*/
class U1QNVal : public QNVal {
public:
  U1QNVal(const int val) : val_(val) {}
  U1QNVal(void) : U1QNVal(0) {}

  U1QNVal *Clone(void) const override { return new U1QNVal(val_); }

  size_t dim(void) const override { return kDimOfU1Repr; }

  int GetVal(void) const override { return val_; }

  U1QNVal *Minus(void) const override { return new U1QNVal(-val_); }

  void AddAssign(const QNVal *prhs_b) override {
    auto prhs_d = static_cast<const U1QNVal *>(prhs_b);   // Do safe downcasting
    val_ += prhs_d->val_;
  }

  // Override for Hashable base class
  size_t Hash(void) const override { return hasher_(val_); }

  // Override for Streamable base class
  void StreamRead(std::istream &is) override { is >> val_; }
  void StreamWrite(std::ostream &os) const override { os << val_ << std::endl; }

  // Override for Showable base class
  void Show(const size_t indent_level = 0) const override {
    std::cout << IndentPrinter(indent_level) << "QNVal: U(1)" << std::endl;
    std::cout << IndentPrinter(indent_level + 1) << "Representation charge: " << val_ << std::endl;
  }

private:
  int val_;
  std::hash<int> hasher_;
};
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_QN_QNVAL_U1_H */
