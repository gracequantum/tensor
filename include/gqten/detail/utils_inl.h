// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 16:20
* 
* Description: GraceQ/tensor project. Inline utility functions used by template headers.
*/
#ifndef GQTEN_DETAIL_UTILS_INL_H
#define GQTEN_DETAIL_UTILS_INL_H


#include <vector>
#include <cmath>

#include "gqten/detail/consts.h"
#include "gqten/detail/value_t.h"


namespace gqten {


// Calculate Cartesian product.
template<typename T>
T CalcCartProd(T v) {
  T s = {{}};
  for (const auto &u : v) {
    T r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}


// Calculate offset for the effective one dimension array.
inline long CalcEffOneDimArrayOffset(
    const std::vector<long> &coors,
    const long ndim,
    const std::vector<long> &data_offsets) {
  long offset = 0;
  for (long i = 0; i < ndim; ++i) {
    offset += coors[i] * data_offsets[i];
  }
  return offset;
}


inline bool DoubleEq(const double a, const double b) {
  if (std::abs(a-b) < kDoubleEpsilon) {
    return true;
  } else {
    return false;
  }
}


inline bool ArrayEq(
    const double *parray1, const size_t size1,
    const double *parray2, const size_t size2) {
  if (size1 !=  size2) {
    return false;
  }
  for (size_t i = 0; i < size1; ++i) {
    if (!DoubleEq(parray1[i], parray2[i])) {
      return false;
    }
  }
  return true;
}


inline GQTEN_Double drand(void) {
  return GQTEN_Double(rand()) / RAND_MAX;
}


inline GQTEN_Complex zrand(void) {
  return GQTEN_Complex(drand(), drand());
}


inline void Rand(GQTEN_Double &d) {
  d = drand();
}


inline void Rand(GQTEN_Complex &z) {
  z = zrand();
}
} /* gqten */
#endif /* ifndef GQTEN_DETAIL_UTILS_INL_H */
