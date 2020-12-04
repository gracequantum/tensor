// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-02 14:15
* 
* Description: GraceQ/tensor project. High performance LAPACK related functions
* based on MKL.
*/

/**
@file lapack.h
@brief High performance LAPACK related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H


#include "gqten/framework/value_t.h"

#include <assert.h>     // assert

#include "mkl.h"


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


namespace hp_numeric {


inline void MatSVD(
    GQTEN_Double *mat,
    const size_t m, const size_t n,
    GQTEN_Double* &u,
    GQTEN_Double* &s,
    GQTEN_Double* &vt
) {
  auto lda = n;
  size_t ldu, ldvt;
  if (m >= n) {
    ldu = n;
    ldvt = n;
    s = (GQTEN_Double *) malloc(n * sizeof(GQTEN_Double));
    vt = (GQTEN_Double *) malloc((ldvt * n) * sizeof(GQTEN_Double));
  } else {
    ldu = m;
    ldvt = n;
    s = (GQTEN_Double *) malloc(m * sizeof(GQTEN_Double));
    vt = (GQTEN_Double *) malloc((ldvt * m) * sizeof(GQTEN_Double));
  }
  u = (GQTEN_Double *) malloc((ldu * m) * sizeof(GQTEN_Double));
  auto info = LAPACKE_dgesdd(
                  LAPACK_ROW_MAJOR, 'S',
                  m, n,
                  mat, lda,
                  s,
                  u, ldu,
                  vt, ldvt
              );
  assert(info == 0);
}


inline void MatSVD(
    GQTEN_Complex *mat,
    const size_t m, const size_t n,
    GQTEN_Complex* &u,
    GQTEN_Double*  &s,
    GQTEN_Complex* &vt
) {
  auto lda = n;
  size_t ldu, ldvt;
  if (m >= n) {
    ldu = n;
    ldvt = n;
    s = (GQTEN_Double *) malloc(n * sizeof(GQTEN_Double));
    vt = (GQTEN_Complex *) malloc((ldvt * n) * sizeof(GQTEN_Complex));
  } else {
    ldu = m;
    ldvt = n;
    s = (GQTEN_Double *) malloc(m * sizeof(GQTEN_Double));
    vt = (GQTEN_Complex *) malloc((ldvt * m) * sizeof(GQTEN_Complex));
  }
  u = (GQTEN_Complex *) malloc((ldu * m) * sizeof(GQTEN_Complex));
  auto info = LAPACKE_zgesdd(
                  LAPACK_ROW_MAJOR, 'S',
                  m, n,
                  mat, lda,
                  s,
                  u, ldu,
                  vt, ldvt
              );
  assert(info == 0);
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H */
