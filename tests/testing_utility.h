// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-30 14:40
*
* Description: GraceQ/tensor project. Utilities for unit tests.
*/
#ifndef TESTS_TESTING_UTILITY_H
#define TESTS_TESTING_UTILITY_H


#include "gqten/framework/value_t.h"    // CoorsT

#include <random>

#include "gtest/gtest.h"


const double kEpsilon = 1.0E-13;


template <typename IntT>
inline IntT RandomInteger(const IntT min, const IntT max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<IntT> distrib(min, max);
  return distrib(gen);
}


inline int RandInt(const int min, const int max) {
  return RandomInteger(min, max);
}


inline size_t RandUnsignedInt(const size_t min, const size_t max) {
  return RandomInteger(min, max);
}


inline size_t RandUnsignedInt(const size_t max) {
  return RandUnsignedInt(0, max);
}


inline gqten::CoorsT TransCoors(
    const gqten::CoorsT &old_coors, const std::vector<size_t> &axes_map) {
  gqten::CoorsT new_coors(old_coors.size());
  for (size_t i = 0; i < axes_map.size(); ++i) {
    new_coors[i] = old_coors[axes_map[i]];
  }
  return new_coors;
}


inline void EXPECT_COMPLEX_EQ(
    const gqten::GQTEN_Complex &lhs,
    const gqten::GQTEN_Complex &rhs) {
  EXPECT_DOUBLE_EQ(lhs.real(), rhs.real());
  EXPECT_DOUBLE_EQ(lhs.imag(), rhs.imag());
}


inline void GtestNear(
    const double lhs,
    const double rhs,
    const double delta) {
  EXPECT_NEAR(lhs, rhs, delta);
}


inline void GtestNear(
    const gqten::GQTEN_Complex lhs,
    const gqten::GQTEN_Complex rhs,
    const double delta) {
  EXPECT_NEAR(lhs.real(), rhs.real(), delta);
  EXPECT_NEAR(lhs.imag(), rhs.imag(), delta);
}


template <typename T>
inline void GtestExpectNear(
    const T lhs,
    const T rhs,
    const double delta) {
  GtestNear(lhs, rhs, delta);
}


inline void GtestArrayEq(const double *lhs, const double *rhs, const long len) {
  for (long i = 0; i < len; ++i) {
    EXPECT_DOUBLE_EQ(lhs[i], rhs[i]);
  }
}


inline void GtestArrayEq(
    const gqten::GQTEN_Complex *lhs, const gqten::GQTEN_Complex *rhs, const long len) {
  for (long i = 0; i < len; ++i) {
    EXPECT_COMPLEX_EQ(lhs[i], rhs[i]);
  }
}
#endif /* ifndef TESTS_TESTING_UTILITY_H */
