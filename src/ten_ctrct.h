// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-09 10:57
* 
* Description: GraceQ/tensor project. Intra-used classes/functions for tensor contraction.
*/
#ifndef GQTEN_TEN_CTRCT_H
#define GQTEN_TEN_CTRCT_H


#include "gqten/gqten.h"

#include <vector>

#include "mkl.h"


namespace gqten {


std::vector<QNBlock *> BlocksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const double,
    const std::vector<QNBlock *> &, const std::vector<QNBlock *> &);

GQTensor *InitCtrctedTen(
    const GQTensor &, const GQTensor &,
    const std::vector<long> &, const std::vector<long> &);

void WrapCtrctBlocks(std::vector<QNBlock *> &, GQTensor *);

std::vector<QNBlock *> MergeCtrctBlks(const std::vector<QNBlock *> &);

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

void CalcBlkCtrctDimsInfo(
    const std::size_t, const QNBlock *, const std::vector<long> &,
    long *, long *);

std::vector<const QNSector *> GetPNewBlkQNScts(
    const QNBlock *, const QNBlock *,
    const std::vector<long> &, const std::vector<long> &);

bool CtrctTransChecker(
    const std::vector<long> &, const long, const char, std::vector<long> &);

std::vector<std::size_t> GenBlksPartHashTable(
    const std::vector<QNBlock *> &, const std::vector<long> &);

inline void FreeBlks(std::vector<QNBlock *> &blks) {
  for (auto &blk : blks) { delete blk; }
}
} /* gqten */ 
#endif /* ifndef GQTEN_TEN_CTRCT_H */
