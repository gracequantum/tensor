// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-21 15:27
*
* Description: High performance tensor transpose function based on HPTT library.
*/

/**
@file ten_trans.h
@brief High performance tensor transpose function based on HPTT library.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H


#include <vector>     // vector


#include "hptt.h"


namespace gqten {


/// High performance numerical functions.
namespace hp_numeric {


const int kTensorTransposeDefaultNumThreads = 4;


int tensor_transpose_num_threads = kTensorTransposeDefaultNumThreads;


int TensorTransposeNumThreads(void) {
  return tensor_transpose_num_threads;
}


void SetTensorTransposeNumThreads(const int num_threads) {
  tensor_transpose_num_threads = num_threads;
}


template <typename ElemT>
void TensorTranspose(
    const std::vector<size_t> &transed_order,
    const size_t ten_rank,
    ElemT *original_data,
    const std::vector<size_t> &original_shape,
    ElemT *transed_data,
    const std::vector<size_t> &transed_shape
) {
  int dim = ten_rank;
  int perm[dim]; for (int i = 0; i < dim; ++i) { perm[i] = transed_order[i]; }
  int sizeA[dim]; for (int i = 0; i < dim; ++i) { sizeA[i] = original_shape[i]; }
  int outerSizeB[dim]; for (int i = 0; i < dim; ++i) { outerSizeB[i] = transed_shape[i]; }
  auto tentrans_plan = hptt::create_plan(perm, dim,
      1.0, original_data, sizeA, sizeA,
      0.0, transed_data, outerSizeB,
      hptt::ESTIMATE,
      tensor_transpose_num_threads, {},
      true
  );
  tentrans_plan->execute();
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_TEN_TRANS_H */
