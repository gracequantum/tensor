// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-01 11:40
*
* Description: GraceQ/tensor project. Matrix representation used by tensor
* decomposition for data blocks in block sparse data tensor.
*/

/**
@file data_blk_mat_repr_info.h
@brief Matrix representation used by tensor decomposition for data blocks in
       block sparse data tensor.
*/
#ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_MAT_H
#define GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_MAT_H


#include "gqten/framework/value_t.h"

#include <vector>     // vector
#include <map>        // map

#include <stddef.h>     // size_t


namespace gqten {


// (part_blk_coors, offset, dim)
using DataBlkMatSector = std::tuple<CoorsT, size_t, size_t>;
// (blk_coors_in_data_blk_mat, blk_idx_in_bsdt)
using DataBlkMatElem = std::pair<CoorsT, size_t>;

using DataBlkMatSectorVec = std::vector<DataBlkMatSector>;
using DataBlkMatElemVec = std::vector<DataBlkMatElem>;


template <typename QNT>
struct TenDecompDataBlkMat {
  QNT lqnflux;
  QNT rqnflux;
  QNT mid_qn;

  size_t rows = 0;
  size_t cols = 0;

  DataBlkMatSectorVec row_scts;
  DataBlkMatSectorVec col_scts;
  DataBlkMatElemVec elems;

  TenDecompDataBlkMat(void) = default;

  TenDecompDataBlkMat(
      const QNT &lqnflux,
      const QNT &rqnflux,
      const QNT &mid_qn
  ) : lqnflux(lqnflux), rqnflux(rqnflux), mid_qn(mid_qn) {}

  bool CheckSame(const QNT &lqnflux_2, const QNT &rqnflux_2) const {
    return (lqnflux_2 == lqnflux) && (rqnflux_2 == rqnflux);
  }

  void InsertElem(
      const size_t m, const size_t n,
      const size_t bsdt_blk_idx,
      const CoorsT &lpart_blk_coors, const CoorsT &rpart_blk_coors
  ) {
    auto i = FindAndSetRowSct_(lpart_blk_coors, m);
    auto j = FindAndSetColSct_(rpart_blk_coors, n);
    elems.push_back(std::make_pair(CoorsT({i, j}), bsdt_blk_idx));
  }

private:
  size_t FindAndSetRowSct_(const CoorsT &lpart_blk_coors, const size_t dim) {
    size_t i = 0;
    size_t offset = 0;
    for (auto &sct : row_scts) {
      if (std::get<0>(sct) == lpart_blk_coors) { return i; }
      i++;
      offset += std::get<2>(sct);
    }
    row_scts.push_back(std::make_tuple(lpart_blk_coors, offset, dim));
    rows += dim;
    return i;
  }

  size_t FindAndSetColSct_(const CoorsT &rpart_blk_coors, const size_t dim) {
    size_t j = 0;
    size_t offset = 0;
    for (auto &sct : col_scts) {
      if (std::get<0>(sct) == rpart_blk_coors) { return j; }
      j++;
      offset += std::get<2>(sct);
    }
    col_scts.push_back(std::make_tuple(rpart_blk_coors, offset, dim));
    cols += dim;
    return j;
  }
};


template <typename QNT>
using IdxDataBlkMatMap = std::map<size_t, TenDecompDataBlkMat<QNT>>;


template <typename QNT>
void InsertDataBlkMat(
    const QNT &lqnflux, const QNT &rqnflux, const QNT &mid_qn,
    const size_t m, const size_t n,
    const size_t bsdt_blk_idx,
    const CoorsT &lpart_blk_coors, const CoorsT &rpart_blk_coors,
    IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map
) {
  for (auto &idx_data_blk_mat : idx_data_blk_mat_map) {
    if (idx_data_blk_mat.second.CheckSame(lqnflux, rqnflux)) {
      idx_data_blk_mat.second.InsertElem(
          m, n,
          bsdt_blk_idx, lpart_blk_coors, rpart_blk_coors
      );
      return;
    }
  }
  TenDecompDataBlkMat<QNT> new_data_blk_mat(lqnflux, rqnflux, mid_qn);
  new_data_blk_mat.InsertElem(
      m, n,
      bsdt_blk_idx, lpart_blk_coors, rpart_blk_coors
  );
  auto new_idx = idx_data_blk_mat_map.size();
  idx_data_blk_mat_map[new_idx] = new_data_blk_mat;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_MAT_H */
