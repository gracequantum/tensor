// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-06 12:09
*
* Description: GraceQ/tensor project. Combine two indexes by generating an index
* combiner.
*/

/**
@file index_combine.h
@brief Combine two indexes by generating an index combiner.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_INDEX_COMBINE_H
#define GQTEN_TENSOR_MANIPULATION_INDEX_COMBINE_H


#include "gqten/gqtensor_all.h"

#include <map>      // map
#include <tuple>    // tuple

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>   // assert


namespace gqten {


// (new_qn, new_qn_dgnc)
template <typename QNT>
using QNDgnc = std::pair<QNT, size_t>;

// (qnsct_index_in_idx1, qnsct_idx_in_idx2, qn_dgnc_index, offset_in_new_qnsct)
using QNSctsOffsetInfo = std::tuple<size_t, size_t, size_t, size_t>;


template <typename IndexT>
inline std::vector<size_t> CalcQNSctDimOffsets(const IndexT &index) {
  std::vector<size_t> qnsct_dim_offsets;
  auto qnsct_num = index.GetQNSctNum();
  qnsct_dim_offsets.reserve(qnsct_num);
  size_t offset = 0;
  for (size_t i = 0; i < qnsct_num; ++i) {
    qnsct_dim_offsets.push_back(offset);
    offset += index.GetQNSct(i).dim();
  }
  return qnsct_dim_offsets;
}


/**
Generate Index combiner.

@param idx1 The first index to be combined.
@param idx2 The second index to be combined.
@param new_idx_dir The direction of the combined index.

@return The index combiner. The index combiner is a tensor initialized as
        GQTensor<TenElemT, QNT>(
            {InverseIndex(idx1),
             InverseIndex(idx2),
             new_idx}
        )
        and only elements corresponding to index combination equal to 1.
*/
template <typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> IndexCombine(
    const Index<QNT> &idx1,
    const Index<QNT> &idx2,
    const GQTenIndexDirType &new_idx_dir = GQTenIndexDirType::OUT
) {
  assert(idx1.GetDir() != GQTenIndexDirType::NDIR);
  assert(idx2.GetDir() != GQTenIndexDirType::NDIR);
  assert(new_idx_dir != GQTenIndexDirType::NDIR);
  std::vector<QNSctsOffsetInfo> qnscts_offset_info_list;
  auto idx1_qnsct_num = idx1.GetQNSctNum();
  auto idx2_qnsct_num = idx2.GetQNSctNum();
  qnscts_offset_info_list.reserve(idx1_qnsct_num * idx2_qnsct_num);
  std::vector<QNDgnc<QNT>> new_qn_dgnc_list;
  for (size_t i = 0; i < idx1_qnsct_num; ++i) {
    for (size_t j = 0; j < idx2_qnsct_num; ++j) {
      auto qnsct_from_idx1 = idx1.GetQNSct(i);
      auto qnsct_from_idx2 = idx2.GetQNSct(j);
      auto qn_from_idx1 = qnsct_from_idx1.GetQn();
      auto qn_from_idx2 = qnsct_from_idx2.GetQn();
      auto dgnc_from_idx1 = qnsct_from_idx1.GetDegeneracy();
      auto dgnc_from_idx2 = qnsct_from_idx2.GetDegeneracy();
      QNT combined_qn;
      switch (idx1.GetDir()) {
        case GQTenIndexDirType::IN:
          switch (idx2.GetDir()) {
            case GQTenIndexDirType::IN:
              combined_qn = qn_from_idx1 + qn_from_idx2;
              break;
            case GQTenIndexDirType::OUT:
              combined_qn = qn_from_idx1 - qn_from_idx2;
              break;
            default:
              assert(false);
          }
          break;
        case GQTenIndexDirType::OUT:
          switch (idx2.GetDir()) {
            case GQTenIndexDirType::IN:
              combined_qn = qn_from_idx2 - qn_from_idx1;
              break;
            case GQTenIndexDirType::OUT:
              combined_qn = (-qn_from_idx2) - qn_from_idx1;
              break;
            default:
              assert(false);
          }
          break;
        default:
          assert(false);
      }
      if (new_idx_dir == GQTenIndexDirType::IN) {
        combined_qn = -combined_qn;
      }
      auto poss_it = std::find_if(
                         new_qn_dgnc_list.begin(),
                         new_qn_dgnc_list.end(),
                         [&combined_qn](const QNDgnc<QNT> &qn_dgnc)->bool {
                           return qn_dgnc.first == combined_qn;
                         }
                     );
      if (poss_it != new_qn_dgnc_list.end()) {
        size_t offset = poss_it->second;
        poss_it->second += (dgnc_from_idx1 * dgnc_from_idx2);
        size_t qn_dgnc_idx = poss_it - new_qn_dgnc_list.begin();
        qnscts_offset_info_list.emplace_back(
            std::make_tuple(i, j, qn_dgnc_idx, offset)
        );
      } else {
        size_t qn_dgnc_idx = new_qn_dgnc_list.size();
        new_qn_dgnc_list.push_back(
            std::make_pair(combined_qn, dgnc_from_idx1 * dgnc_from_idx2)
        );
        qnscts_offset_info_list.emplace_back(
            std::make_tuple(i, j, qn_dgnc_idx, 0)
        );
      }
    }
  }
  QNSectorVec<QNT> qnscts;
  std::vector<size_t> new_idx_qnsct_dim_offsets;
  qnscts.reserve(new_qn_dgnc_list.size());
  new_idx_qnsct_dim_offsets.reserve(new_qn_dgnc_list.size());
  size_t qnsct_dim_offset = 0;
  for (auto &new_qn_dgnc : new_qn_dgnc_list) {
    qnscts.push_back(QNSector<QNT>(new_qn_dgnc.first, new_qn_dgnc.second));
    new_idx_qnsct_dim_offsets.push_back(qnsct_dim_offset);
    qnsct_dim_offset += new_qn_dgnc.second;
  }
  Index<QNT> new_idx(qnscts, new_idx_dir);

  GQTensor<TenElemT, QNT> index_combiner({idx1, idx2, new_idx});
  std::vector< CoorsT > blk_coors_s;
  blk_coors_s.reserve(qnscts_offset_info_list.size() );

  for (auto &qnscts_offset_info : qnscts_offset_info_list) {
    size_t qnsct_coor_from_idx1 = std::get<0>(qnscts_offset_info);
    size_t qnsct_coor_from_idx2 = std::get<1>(qnscts_offset_info);
    size_t qnsct_coor_from_new_idx = std::get<2>(qnscts_offset_info);
    blk_coors_s.push_back({qnsct_coor_from_idx1, qnsct_coor_from_idx2, qnsct_coor_from_new_idx} );
  }
  auto& blk_spar_data_ten = index_combiner.GetBlkSparDataTen();
  blk_spar_data_ten.DataBlksInsert(blk_coors_s,
                                   true,
                                   true);
  auto idx1_qnsct_dim_offsets = CalcQNSctDimOffsets(idx1);
  auto idx2_qnsct_dim_offsets = CalcQNSctDimOffsets(idx2);
  for (auto &qnscts_offset_info : qnscts_offset_info_list) {
    auto qnsct_coor_from_idx1 = std::get<0>(qnscts_offset_info);
    auto qnsct_coor_from_idx2 = std::get<1>(qnscts_offset_info);
    auto qnsct_coor_from_new_idx = std::get<2>(qnscts_offset_info);
    auto intra_offset_in_new_qnsct = std::get<3>(qnscts_offset_info);
    auto qnsct_from_idx1 = idx1.GetQNSct(qnsct_coor_from_idx1);
    auto qnsct_from_idx2 = idx2.GetQNSct(qnsct_coor_from_idx2);
    size_t k = new_idx_qnsct_dim_offsets[
                   qnsct_coor_from_new_idx
               ] + intra_offset_in_new_qnsct;
    for (size_t i = 0; i < qnsct_from_idx1.GetDegeneracy(); ++i) {
      for (size_t j = 0; j < qnsct_from_idx2.GetDegeneracy(); ++j) {
        index_combiner.SetElem(
            {
                idx1_qnsct_dim_offsets[qnsct_coor_from_idx1] + i,
                idx2_qnsct_dim_offsets[qnsct_coor_from_idx2] + j,
                k
            },
            1.0
        );
        k++;
      }
    }
  }
  return index_combiner;
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_INDEX_COMBINE_H */
