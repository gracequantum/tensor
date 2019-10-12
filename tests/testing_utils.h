// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-29 16:41
* 
* Description: GraceQ/tensor project. Testing utilities.
*/
#ifndef GQTEN_TESTING_UTILS_H
#define GQTEN_TESTING_UTILS_H


#include <vector>

#include "gtest/gtest.h"

#include "gqten/detail/value_t.h"

#include "mkl.h"    // Included after other header file. Because GraceQ needs redefine MKL_Complex16 to gqten::GQTEN_Complex .


using namespace gqten;


const double kEpsilon = 1.0E-12;


inline std::vector<long> TransCoors(
    const std::vector<long> &old_coors, const std::vector<long> &axes_map) {
  std::vector<long> new_coors(old_coors.size());
  for (std::size_t i = 0; i < axes_map.size(); ++i) {
    new_coors[i] = old_coors[axes_map[i]];
  }
  return new_coors;
}


inline void EXPECT_COMPLEX_EQ(
    const GQTEN_Complex &lhs,
    const GQTEN_Complex &rhs) {
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
    const GQTEN_Complex lhs,
    const GQTEN_Complex rhs,
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
    const GQTEN_Complex *lhs, const GQTEN_Complex *rhs, const long len) {
  for (long i = 0; i < len; ++i) {
    EXPECT_COMPLEX_EQ(lhs[i], rhs[i]);
  }
}


inline void CblasGemm(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const MKL_INT m, const MKL_INT n, const MKL_INT k,
    const GQTEN_Double alpha,
    const GQTEN_Double *a, const MKL_INT lda,
    const GQTEN_Double *b, const MKL_INT ldb,
    const GQTEN_Double beta,
    GQTEN_Double *c, const MKL_INT ldc) {
  cblas_dgemm(
      Layout,
      transa, transb,
      m, n, k,
      alpha,
      a, lda,
      b, ldb,
      beta,
      c, ldc);
}


inline void CblasGemm(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const MKL_INT m, const MKL_INT n, const MKL_INT k,
    const GQTEN_Complex alpha,
    const GQTEN_Complex *a, const MKL_INT lda,
    const GQTEN_Complex *b, const MKL_INT ldb,
    const GQTEN_Complex beta,
    GQTEN_Complex *c, const MKL_INT ldc) {
  cblas_zgemm(
      Layout,
      transa, transb,
      m, n, k,
      &alpha,
      a, lda,
      b, ldb,
      &beta,
      c, ldc);
}
#endif /* ifndef GQTEN_TESTING_UTILS_H */
