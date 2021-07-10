// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-07-06 14:10
*
* Description: GraceQ/tensor project. Basic common stuffs for tensor decomposition.
*/

/**
@file ten_decomp_basic.h
@brief Basic common stuffs for tensor decomposition.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_DECOMP_BASIC_H
#define GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_DECOMP_BASIC_H


#include "gqten/gqtensor_all.h"
#include "gqten/gqtensor/blk_spar_data_ten/data_blk_mat.h"    // TenDecompDataBlkMat, IdxDataBlkMatMap


namespace gqten {


inline void CalcDataBlkMatShape(
    const ShapeT &shape, const size_t ldims,
    size_t &rows, size_t &cols
) {
  size_t m = 1;
  size_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i < ldims) {
      m *= shape[i];
    } else {
      n *= shape[i];
    }
  }
  rows = m;
  cols = n;
}


template <typename QNT>
QNT CalcDataBlkMatLeftQNFlux(
    const CoorsT &blk_coors,
    const size_t ldims,
    const IndexVec<QNT> &indexes
) {
  QNT lqnflux;
  if (indexes[0].GetDir() == GQTenIndexDirType::OUT) {
    lqnflux = indexes[0].GetQNSct(blk_coors[0]).GetQn();
  } else if (indexes[0].GetDir() == GQTenIndexDirType::IN) {
    lqnflux = -indexes[0].GetQNSct(blk_coors[0]).GetQn();
  }
  if (ldims == 1) {
    return lqnflux;
  } else {
    for (size_t i = 1; i < ldims; ++i) {
      if (indexes[i].GetDir() == GQTenIndexDirType::OUT) {
        lqnflux += indexes[i].GetQNSct(blk_coors[i]).GetQn();
      } else if (indexes[i].GetDir() == GQTenIndexDirType::IN) {
        lqnflux += (-indexes[i].GetQNSct(blk_coors[i]).GetQn());    // TODO: Implement -= for QN!
      }
    }
    return lqnflux;
  }
}


template <typename TenElemT, typename QNT>
IdxDataBlkMatMap<QNT> GenIdxTenDecompDataBlkMats(
    const GQTensor<TenElemT, QNT> &t,
    const size_t ldims,
    const QNT &lqndiv
) {
  auto t_blk_idx_data_blk_map = t.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  auto t_qndiv = Div(t);
  IdxDataBlkMatMap<QNT> idx_data_blk_mat_map;
  for (auto &blk_idx_data_blk : t_blk_idx_data_blk_map) {
    auto blk_idx = blk_idx_data_blk.first;
    auto data_blk = blk_idx_data_blk.second;
    size_t m, n;
    CalcDataBlkMatShape(data_blk.shape, ldims, m, n);
    auto bsdt_blk_coors = data_blk.blk_coors;
    CoorsT lpart_blk_coors(
        bsdt_blk_coors.begin(),
        bsdt_blk_coors.begin() + ldims
    );
    CoorsT rpart_blk_coors(
        bsdt_blk_coors.begin() + ldims,
        bsdt_blk_coors.end()
    );
    auto lqnflux = CalcDataBlkMatLeftQNFlux(
                       data_blk.blk_coors,
                       ldims,
                       t.GetIndexes()
                   );
    auto rqnflux = t_qndiv - lqnflux;
    auto mid_qn = lqndiv - lqnflux;
    InsertDataBlkMat(
        lqnflux, rqnflux, mid_qn,
        m, n,
        blk_idx, lpart_blk_coors, rpart_blk_coors,
        idx_data_blk_mat_map
    );
  }
  return idx_data_blk_mat_map;
}
} /* gqten */ 
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_DECOMP_BASIC_H */
