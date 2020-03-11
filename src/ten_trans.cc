// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 19:10
* 
* Description: GraceQ/tensor project. Implement dense tensor transpose using hptt library.
*/
#include "gqten/gqten.h"
#include "ten_trans.h"

#include <vector>

#include "hptt.h"


namespace gqten {


int tensor_transpose_num_threads = kTensorTransposeDefaultNumThreads;


int GQTenGetTensorTransposeNumThreads(void) {
  return tensor_transpose_num_threads;
}


void GQTenSetTensorTransposeNumThreads(const int num_threads) {
  tensor_transpose_num_threads = num_threads;
}


GQTEN_Double *DenseTensorTranspose(
    const GQTEN_Double *old_data,
    const long old_ndim,
    const long old_size,
    const std::vector<long> &old_shape,
    const std::vector<long> &transed_axes) {
  int dim = old_ndim;
  int perm[dim];  for (int i = 0; i < dim; ++i) { perm[i] = transed_axes[i]; }
  int sizeA[dim]; for (int i = 0; i < dim; ++i) { sizeA[i] = old_shape[i]; }
  int outerSizeB[dim];
  for (int i = 0; i < dim; ++i) { outerSizeB[i] = old_shape[perm[i]]; }
  //auto transed_data = new GQTEN_Double[old_size];
  auto transed_data = (GQTEN_Double *)malloc(old_size * sizeof(GQTEN_Double));
  dTensorTranspose(perm, dim,
      1.0, old_data, sizeA, sizeA,
      0.0, transed_data, outerSizeB,
      tensor_transpose_num_threads, 1);
  return transed_data;
}


GQTEN_Complex *DenseTensorTranspose(
    const GQTEN_Complex *old_data,
    const long old_ndim,
    const long old_size,
    const std::vector<long> &old_shape,
    const std::vector<long> &transed_axes) {
  int dim = old_ndim;
  int perm[dim];  for (int i = 0; i < dim; ++i) { perm[i] = transed_axes[i]; }
  int sizeA[dim]; for (int i = 0; i < dim; ++i) { sizeA[i] = old_shape[i]; }
  int outerSizeB[dim];
  for (int i = 0; i < dim; ++i) { outerSizeB[i] = old_shape[perm[i]]; }
  //auto transed_data = new GQTEN_Complex[old_size];
  auto transed_data = (GQTEN_Complex *)malloc(old_size * sizeof(GQTEN_Complex));
  auto tentrans_plan = hptt::create_plan(perm, dim,
      1.0, old_data, sizeA, sizeA,
      0.0, transed_data, outerSizeB,
      hptt::ESTIMATE,
      tensor_transpose_num_threads, {},
      true);
  tentrans_plan->execute();
  return transed_data;
}
} /* gqten */ 
