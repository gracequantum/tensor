// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-17 13:27
* 
* Description: GraceQ/tensor project. Inline functions for implementing tensor contraction.
*/
#ifndef GQTEN_DETAIL_TEN_CTRCT_INL_H
#define GQTEN_DETAIL_TEN_CTRCT_INL_H


#include <iostream>
#include <vector>

#include "mkl.h"

#include "gqten/detail/value_t.h"


namespace gqten {


inline void CblasAxpy(
    const MKL_INT n, const GQTEN_Double a,
    const GQTEN_Double *x, const MKL_INT incx,
    GQTEN_Double *y, const MKL_INT incy) {
  cblas_daxpy(
      n,
      a, x, incx,
      y, incy);
}


inline void CblasAxpy(
    const MKL_INT n, const GQTEN_Complex a,
    const GQTEN_Complex *x, const MKL_INT incx,
    GQTEN_Complex *y, const MKL_INT incy) {
  cblas_zaxpy(
      n,
      &a, x, incx,
      y, incy);
}


inline void GemmBatch(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE *transa_array, const CBLAS_TRANSPOSE *transb_array,
    const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
    const GQTEN_Double *alpha_array,
    const GQTEN_Double **a_array, const MKL_INT *lda_array,
    const GQTEN_Double **b_array, const MKL_INT *ldb_array,
    const GQTEN_Double *beta_array,
    GQTEN_Double **c_array, const MKL_INT *ldc_array,
    const MKL_INT group_count,
    const MKL_INT *group_size) {

#ifdef GQTEN_USE_MKL_GEMM_BATCH

  cblas_dgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      alpha_array,
      a_array, lda_array,
      b_array, ldb_array,
      beta_array,
      c_array, ldc_array,
      group_count,
      group_size);

#else // Use direct gemm loop.

  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < group_size[i]; ++j) {
      cblas_dgemm(
          Layout,
          transa_array[i], transb_array[i],
          m_array[i], n_array[i], k_array[i],
          alpha_array[i],
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
    }
  }

#endif
}


inline void GemmBatch(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE *transa_array, const CBLAS_TRANSPOSE *transb_array,
    const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
    const GQTEN_Complex *alpha_array,
    const GQTEN_Complex **a_array, const MKL_INT *lda_array,
    const GQTEN_Complex **b_array, const MKL_INT *ldb_array,
    const GQTEN_Complex *beta_array,
    GQTEN_Complex **c_array, const MKL_INT *ldc_array,
    const MKL_INT group_count,
    const MKL_INT *group_size) {

#ifdef GQTEN_USE_MKL_GEMM_BATCH

  cblas_zgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      &alpha_array,
      a_array, lda_array,
      b_array, ldb_array,
      &beta_array,
      c_array, ldc_array,
      group_count,
      group_size);

#else // Use direct gemm loop.

  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < group_size[i]; ++j) {
      cblas_zgemm(
          Layout,
          transa_array[i], transb_array[i],
          m_array[i], n_array[i], k_array[i],
          &alpha_array[i],
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          &beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
    }
  }

#endif
}
} /* gqten */ 
#endif /* ifndef GQTEN_DETAIL_TEN_CTRCT_INL_H */
