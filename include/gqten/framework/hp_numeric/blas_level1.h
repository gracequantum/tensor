// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-24 20:13
*
* Description: High performance BLAS Level 1 related functions based on MKL.
*/

/**
@file blas_level1.h
@brief High performance BLAS Level 1 related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H


#include "gqten/framework/value_t.h"      // GQTEN_Double, GQTEN_Complex

#include "mkl.h"      // cblas_*axpy


namespace gqten {


/// High performance numerical functions.
namespace hp_numeric {


inline void VectorAddTo(
    const GQTEN_Double *x,
    const size_t size,
    GQTEN_Double *y
) {
  cblas_daxpy(size, 1.0, x, 1, y, 1);
}


inline void VectorAddTo(
    const GQTEN_Complex *x,
    const size_t size,
    GQTEN_Complex *y
) {
  GQTEN_Complex a(1.0);
  cblas_zaxpy(size, &a, x, 1, y, 1);
}


inline void VectorScale(
    GQTEN_Double *x,
    const size_t size,
    const GQTEN_Double a
) {
  cblas_dscal(size, a, x, 1);
}


inline void VectorScale(
    GQTEN_Complex *x,
    const size_t size,
    const GQTEN_Complex a
) {
  cblas_zscal(size, &a, x, 1);
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H */
