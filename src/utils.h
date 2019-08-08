// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 17:07
* 
* Description: GraceQ/tensor project. Forward declaration for intra-used utility functions.
*/
#ifndef GQTEN_UTILS_H
#define GQTEN_UTILS_H


#include "gqten/gqten.h"

#include <vector>


namespace gqten {


QN CalcDiv(const QNSectorSet &, const std::vector<Index> &);

QN CalcDiv(const std::vector<QNSector> &, const std::vector<Index> &);

std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &);

long MulToEnd(const std::vector<long> &, int);

std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &);

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


// Inline functions.
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


} /* gqten */ 
#endif /* ifndef GQTEN_UTILS_H */
