// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-27 09:21
*
* Description: GraceQ/tensor project. Basic tensor operations.
*/

/**
@file basic_operations.h
@brief Basic tensor operations.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H
#define GQTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H


#include "gqten/gqtensor/gqtensor.h"    // GQTensor

#include <iostream>     // cout, endl
#include <iterator>     // next

#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


/**
Calculate dagger of a GQTensor.

@param t A GQTensor \f$ T \f$.

@return Daggered \f$ T^{\dagger} \f$.
*/
template <typename GQTensorT>
GQTensorT Dag(const GQTensorT &t) {
  GQTensorT t_dag(t);
  t_dag.Dag();
  return t_dag;
}


/**
Calculate the quantum number divergence of a GQTensor.

@param t A GQTensor.

@return The quantum number divergence.
*/
template <typename ElemT, typename QNT>
QNT Div(const GQTensor<ElemT, QNT> &t) {
  assert(!t.IsDefault());
  if (t.IsScalar()) {
    std::cout << "Tensor is a scalar. Return empty quantum number."
              << std::endl;
    return QNT();
  } else {
    auto qnblk_num = t.GetQNBlkNum();
    if (qnblk_num == 0) {
      std::cout << "Tensor does not have a block. Return empty quantum number."
                << std::endl;
      return QNT();
    } else {
      auto blk_idx_data_blk_map = t.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
      auto indexes = t.GetIndexes();
      auto first_blk_idx_data_blk = blk_idx_data_blk_map.begin();
      auto div = CalcDiv(indexes, first_blk_idx_data_blk->second.blk_coors);
      for (
          auto it = std::next(first_blk_idx_data_blk);
          it != blk_idx_data_blk_map.end();
          ++it
      ) {
        auto blki_div = CalcDiv(indexes, it->second.blk_coors);
        if (blki_div != div) {
          std::cout << "Tensor does not have a special divergence. Return empty quantum number."
                    << std::endl;
          return QNT();
        }
      }
      return div;
    }
  }
}


/**
Convert a real GQTensor to a complex GQTensor.
*/
template <typename QNT>
GQTensor<GQTEN_Complex, QNT> ToComplex(
    const GQTensor<GQTEN_Double, QNT> &real_t
) {
  assert(!real_t.IsDefault());
  GQTensor<GQTEN_Complex, QNT> cplx_t(real_t.GetIndexes());
  if (cplx_t.IsScalar()) {
    cplx_t.SetElem({}, real_t.GetElem({}));
  } else {
    cplx_t.GetBlkSparDataTen().CopyFromReal(real_t.GetBlkSparDataTen());
  }
  return cplx_t;
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H */
