// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-28 19:07
*
* Description: GraceQ/tensor project. High performance BLAS Level 3 related
* functions based on MKL.
*/

/**
@file blas_level3.h
@brief High performance BLAS Level 3 related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H


#include "gqten/framework/value_t.h"      // GQTEN_Double, GQTEN_Complex

#include <assert.h>     // assert

#include "mkl.h"      //cblas_*gemm


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {

/// High performance numerical functions.
namespace hp_numeric {


inline void MatMultiply(
    const GQTEN_Double *a,
    const GQTEN_Double *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const GQTEN_Double beta,
    GQTEN_Double *c) {
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1.0,
      a, k,
      b, n,
      beta,
      c, n
  );
}


inline void MatMultiply(
    const GQTEN_Complex *a,
    const GQTEN_Complex *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const GQTEN_Complex beta,
    GQTEN_Complex *c) {
  GQTEN_Complex alpha(1.0);
  cblas_zgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      &alpha,
      a, k,
      b, n,
      &beta,
      c, n
  );
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H */
