// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-17 13:27
* 
* Description: GraceQ/tensor project. Inline functions for implementing tensor contraction.
*/
#ifndef GQTEN_DETAIL_TEN_CTRCT_INL_H
#define GQTEN_DETAIL_TEN_CTRCT_INL_H


#include <vector>

#include "mkl.h"


namespace gqten {


inline void GemmBatch(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE *transa_array, const CBLAS_TRANSPOSE *transb_array,
    const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
    const double *alpha_array,
    const double **a_array, const MKL_INT *lda_array,
    const double **b_array, const MKL_INT *ldb_array,
    const double *beta_array,
    double **c_array, const MKL_INT *ldc_array,
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


template <typename TenElemType>
bool CtrctTransChecker(
    const std::vector<long> &ctrct_axes,
    const long ndim,
    const char position,
    std::vector<long> &transed_axes) {
  auto ctrct_ndim = ctrct_axes.size();
  std::vector<long> saved_axes(ndim-ctrct_ndim);
  std::size_t saved_axes_idx = 0;
  std::vector<long> ordered_axes(ndim);
  for (long i = 0; i < ndim; ++i) {
    if (std::find(ctrct_axes.begin(), ctrct_axes.end(), i) ==
        ctrct_axes.end()) {
      saved_axes[saved_axes_idx] = i;
      saved_axes_idx++;
    }
    ordered_axes[i] = i;
  }
  switch (position) {
    case 'a':
      transed_axes = saved_axes;
      transed_axes.insert(
          transed_axes.end(),
          ctrct_axes.begin(), ctrct_axes.end());
      if (transed_axes != ordered_axes) { return true; }
      break;
    case 'b':
      transed_axes = ctrct_axes;
      transed_axes.insert(
          transed_axes.end(),
          saved_axes.begin(), saved_axes.end());
      if (transed_axes != ordered_axes) { return true; }
      break;
    default:
      std::cout << "position must be 'a' or 'b', but" << position << std::endl;
      exit(1);
  }
  return false;
}
} /* gqten */ 
#endif /* ifndef GQTEN_DETAIL_TEN_CTRCT_INL_H */
