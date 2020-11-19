// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-16 13:05
*
* Description: GraceQ/tensor project. Information of a quantum number block.
*/

/**
@file qnblk_info.h
@brief Information of a quantum number block.
*/
#ifndef GQTEN_GQTENSOR_QNBLK_INFO_H
#define GQTEN_GQTENSOR_QNBLK_INFO_H


#include "gqten/framework/value_t.h"    // CoorsT, ShapeT
#include "gqten/gqtensor/qnsct.h"       // QNSectorVec


namespace gqten {


/**
Information of a quantum number block.

@tparam QNT Type of the quantum number.
*/
template <typename QNT>
class QNBlkInfo {
public:
  QNBlkInfo(void) = default;

  QNBlkInfo(const QNSectorVec<QNT> &qnscts) : qnscts(qnscts) {}

  /**
  Calculate the shape of the degeneracy space.
  */
  ShapeT CalcDgncSpaceShape(void) {
    ShapeT dgnc_space_shape;
    for (auto &qnsct : qnscts) {
      dgnc_space_shape.push_back(qnsct.GetDegeneracy());
    }
    return dgnc_space_shape;
  }

  size_t PartHash(const CoorsT &) const;
  void Transpose(const CoorsT &);

  QNSectorVec<QNT> qnscts;
};
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_QNBLK_INFO_H */
