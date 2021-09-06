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

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


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
Calculate element-wise complex conjugate of a GQTensor.

@param t A GQTensor \f$ T \f$.

@return \f$ \bar{T} \f$
*/
template <typename GQTensorT>
GQTensorT Conj(const GQTensorT &t) {
  GQTensorT t_conj(t);
  t_conj.Conj();
  return t_conj;
}


/**
Calculate the quantum number divergence of a GQTensor by call GQTensor::Div().

@param t A GQTensor.

@return The quantum number divergence.
*/
template <typename ElemT, typename QNT>
inline QNT Div(const GQTensor<ElemT, QNT> &t) { return t.Div(); }


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

/**
Get the real part of a complex GQTensor.
*/
template <typename QNT>
GQTensor<GQTEN_Double, QNT>
GetReal(const GQTensor<GQTEN_Complex, QNT> &cplx_t) {
  assert(!cplx_t.IsDefault());
  GQTensor<GQTEN_Double, QNT> real_t(cplx_t.GetIndexes());
  if (cplx_t.IsScalar()) {
    real_t.SetElem({}, cplx_t.GetElem({}).real());
  } else {
    real_t.GetBlkSparDataTen().CopyRealFromCplx(cplx_t.GetBlkSparDataTen());
  }
  return real_t;
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_BASIC_OPERATIONS_H */
