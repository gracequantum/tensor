// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 16:20
* 
* Description: GraceQ/tensor project. Inline utility functions used by template headers.
*/
#ifndef GQTEN_IMPL_UTILS_INL_H
#define GQTEN_IMPL_UTILS_INL_H


#include <vector>


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
#endif /* ifndef GQTEN_IMPL_UTILS_INL_H */
