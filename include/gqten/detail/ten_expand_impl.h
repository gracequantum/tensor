// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-06-22 09:40
* 
* Description: GraceQ/tensor project. Implementation details for tensor expansion.
*/
#include "gqten/gqten.h"
#include "gqten/detail/utils_inl.h"

#include <unordered_map>  // unordered_map
#include <algorithm>      // find
#include <iterator>       // distance

#include <assert.h>       // assert

#ifdef Release
  #define NDEBUG
#endif

namespace gqten {


// Forward declarations.
using IndexVec = std::vector<Index>;


class QNSectorHashFunc {
public:
  size_t operator() (const QNSector &qnsct) const {
    return qnsct.Hash(); 
  }
};


using QNSctExpandMap = std::unordered_map<QNSector, QNSector, QNSectorHashFunc>;
using QNSctExpandMapVec = std::vector<QNSctExpandMap>;

enum ExpandDirection {Up, Down};

IndexVec ExpandIdxs(
    const IndexVec &,
    const IndexVec &,
    const std::vector<size_t> &,
    QNSctExpandMapVec &,
    QNSctExpandMapVec &
);

Index ExpandIndex(
    const Index &,
    const Index &,
    QNSctExpandMapVec &,
    QNSctExpandMapVec &
);

template <typename TenElemType>
void ExpandSingleTen(
    const GQTensor<TenElemType> *,
    const std::vector<size_t> &,
    const QNSctExpandMapVec &,
    const ExpandDirection,
    GQTensor<TenElemType> &
);

template <typename TenElemType>
void ExpandSingleBlk(
    const QNBlock<TenElemType> *,
    const std::vector<size_t> &,
    const ExpandDirection,
    QNBlock<TenElemType> &
);

long CalcExpandedOffset(
    const long,
    const std::vector<long> &,
    const std::vector<long> &,
    const std::vector<size_t> &,
    const ExpandDirection
);

std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &);

std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &);

/** Perform tensor expansion.
 */
template <typename TenElemType>
void Expand(
    const GQTensor<TenElemType> *pta,           ///< Pointer to input tensor A.
    const GQTensor<TenElemType> *ptb,           ///< Pointer to input tensor B.
    const std::vector<size_t> expand_idx_nums,  ///< Index numbers which index to be expanded.
    GQTensor<TenElemType> *ptc                  ///< Pointer to expanded tensor C.
) {
  assert(pta->shape.size() == ptb->shape.size());                             // To be expanded tensors should have the same rank.
#ifndef NDEBUG                                                                // Indexes of the to be expanded tensors should have the same directions.
  for (size_t i = 0; i < pta->shape.size(); i++) {
    assert(pta->indexes[i].dir == ptb->indexes[i].dir);
  }
#endif /* ifndef NDEBUG */
  assert(Div(*pta) == Div(*ptb) || Div(*pta) == QN() || Div(*ptb) == QN());   // To be expanded tensors should have the same quantum number divergence or a null quantum number divergence QN().
  // Expand indexes
  QNSctExpandMapVec ta_idx_qnsct_expand_maps, tb_idx_qnsct_expand_maps;
  auto expanded_idxs = ExpandIdxs(
                           pta->indexes,
                           ptb->indexes,
                           expand_idx_nums,
                           ta_idx_qnsct_expand_maps,
                           tb_idx_qnsct_expand_maps
                       );

  // Expand tensor A and B
  //[> TODO: Parallel implementation <]
  GQTensor<TenElemType> ta_expanded(expanded_idxs), tb_expanded(expanded_idxs);
  ExpandSingleTen(
      pta, expand_idx_nums,
      ta_idx_qnsct_expand_maps, Down,
      ta_expanded
  );
  assert(Div(ta_expanded) == Div(*pta));
  ExpandSingleTen(
      ptb, expand_idx_nums,
      tb_idx_qnsct_expand_maps, Up,
      tb_expanded
  );
  assert(Div(tb_expanded) == Div(*ptb));

  // Sum A_expanded and B_expanded to obtain final result
  (*ptc) = ta_expanded + tb_expanded;
}


/** Expand Indexes of tensor A and B.
 */
inline
IndexVec ExpandIdxs(
    const IndexVec &ta_idxs,                       ///< Indexes of tensor A.
    const IndexVec &tb_idxs,                       ///< Indexes of tensor B.
    const std::vector<size_t> &expand_idx_nums,    ///< Index numbers which index to be expanded.
    QNSctExpandMapVec &ta_idx_qnsct_expand_maps,   ///< Mappings from QNSectors of Index of A to corresponding expanded QNSectors of Index of C.
    QNSctExpandMapVec &tb_idx_qnsct_expand_maps    ///< Mappings from QNSectors of Index of B to corresponding expanded QNSectors of Index of C.
) {
  IndexVec expanded_idxs;
  size_t ten_rank = ta_idxs.size();
  for (size_t i = 0; i < ten_rank; ++i) {
    auto poss_it = std::find(expand_idx_nums.cbegin(), expand_idx_nums.cend(), i);
    if (poss_it != expand_idx_nums.cend()) {    // Corresponding Index needs to expand.
      expanded_idxs.push_back(
          ExpandIndex(
              ta_idxs[i],
              tb_idxs[i],
              ta_idx_qnsct_expand_maps,
              tb_idx_qnsct_expand_maps
          )
      );
    } else {                                    // Corresponding Index does not need to expand.
      expanded_idxs.push_back(ta_idxs[i]); 
    }
  }

  return expanded_idxs;
}


inline
Index ExpandIndex(
    const Index &idx_from_ta,
    const Index &idx_from_tb,
    QNSctExpandMapVec &ta_idx_qnsct_expand_maps,
    QNSctExpandMapVec &tb_idx_qnsct_expand_maps
) {
  std::vector<QNSector> expanded_qnscts;
  QNSctExpandMap ta_idx_qnsct_expand_map, tb_idx_qnsct_expand_map;
  auto qnscts_from_ta = idx_from_ta.qnscts;
  auto qnscts_from_tb = idx_from_tb.qnscts;
  for (auto &qnsct_from_ta : qnscts_from_ta) {
    auto qnscts_from_tb_size = qnscts_from_tb.size();
    bool has_matched_qnsct_in_qnscts_from_tb = false;
    for (size_t j = 0; j < qnscts_from_tb_size; ++j) {
      if (qnscts_from_tb[j].qn == qnsct_from_ta.qn) {
        auto expanded_qnsct = QNSector(
                                  qnsct_from_ta.qn,
                                  qnsct_from_ta.dim + qnscts_from_tb[j].dim
                              );
        expanded_qnscts.push_back(expanded_qnsct);
        ta_idx_qnsct_expand_map[qnsct_from_ta] = expanded_qnsct;
        tb_idx_qnsct_expand_map[qnscts_from_tb[j]] = expanded_qnsct;
        qnscts_from_tb.erase(qnscts_from_tb.cbegin() + j);
        has_matched_qnsct_in_qnscts_from_tb = true;
        break;
      }
    }
    if (!has_matched_qnsct_in_qnscts_from_tb) {
      expanded_qnscts.push_back(qnsct_from_ta);
      ta_idx_qnsct_expand_map[qnsct_from_ta] = qnsct_from_ta; 
    }
  }

  for (auto &qnsct_from_tb : qnscts_from_tb) {
    expanded_qnscts.push_back(qnsct_from_tb);
    tb_idx_qnsct_expand_map[qnsct_from_tb] = qnsct_from_tb;
  }

  ta_idx_qnsct_expand_maps.push_back(ta_idx_qnsct_expand_map);
  tb_idx_qnsct_expand_maps.push_back(tb_idx_qnsct_expand_map);
  return Index(expanded_qnscts, idx_from_ta.dir);
}


/** Expand a single tensor.
 */
template <typename TenElemType>
void ExpandSingleTen(
    const GQTensor<TenElemType> *pt,
    const std::vector<size_t> &expand_idx_nums,
    const QNSctExpandMapVec &t_idx_qnsct_expand_maps,
    const ExpandDirection expand_dir,
    GQTensor<TenElemType> &expanded_ten
) {
  auto ten_rank = pt->indexes.size();
  for (auto pblk : pt->cblocks()) {
    std::vector<QNSector> expanded_blk_qnscts;
    for (size_t i = 0; i < ten_rank; ++i) {
      auto poss_it = std::find(
                         expand_idx_nums.cbegin(), expand_idx_nums.cend(), i
                     );
      if (poss_it != expand_idx_nums.cend()) {   // Related dimension is expanded
        auto map_idx = std::distance(expand_idx_nums.cbegin(), poss_it);
        expanded_blk_qnscts.push_back(
            t_idx_qnsct_expand_maps[map_idx].at(pblk->qnscts[i])
        );
      } else {
        expanded_blk_qnscts.push_back(pblk->qnscts[i]);
      }
    }
    QNBlock<TenElemType> *pexpanded_blk = new QNBlock<TenElemType>(
                                              expanded_blk_qnscts
                                          );
    ExpandSingleBlk(pblk, expand_idx_nums, expand_dir, (*pexpanded_blk));
    (expanded_ten.blocks()).push_back(pexpanded_blk);
  }
}


// TODO: performance improvement
template <typename TenElemType>
void ExpandSingleBlk(
    const QNBlock<TenElemType> *pblk,
    const std::vector<size_t> &expand_dim_nums,
    const ExpandDirection expand_dir,
    QNBlock<TenElemType> &expanded_blk) {
  auto blk_cdata = pblk->cdata();
  auto expanded_blk_data = expanded_blk.data();
  auto blk_data_offsets = CalcMultiDimDataOffsets(pblk->shape);
  auto expanded_blk_data_offsets = CalcMultiDimDataOffsets(expanded_blk.shape);
  auto old_blk_coors_set = GenAllCoors(pblk->shape);    // TODO: do not use GenAllCoors
  for (auto &coors : old_blk_coors_set) {
    expanded_blk_data[
        CalcExpandedOffset(
            coors,
            pblk->shape,
            expanded_blk.shape,
            expanded_blk_data_offsets,
            expand_dim_nums,
            expand_dir
        )
    ] = blk_cdata[
            CalcEffOneDimArrayOffset(coors, pblk->ndim, blk_data_offsets)
        ]; 
  }
}


inline
long CalcExpandedOffset(
    const std::vector<long> old_coors,
    const std::vector<long> old_shape,
    const std::vector<long> expanded_shape,
    const std::vector<long> &expanded_data_offsets,
    const std::vector<size_t> &expand_dim_nums,
    const ExpandDirection expand_dir
) {
  size_t blk_ndim = old_coors.size();
  // Calculate expanded coordinates
  std::vector<long> expanded_coors;
  if (expand_dir == Down) {
    expanded_coors = old_coors;
  } else if (expand_dir == Up) {
    for (size_t i = 0; i < blk_ndim; i++) {
      if (
          std::find(expand_dim_nums.cbegin(), expand_dim_nums.cend(), i) !=
          expand_dim_nums.cend()
      ) {   // Related dimension is expanded
        expanded_coors.push_back(expanded_shape[i]-old_shape[i]+old_coors[i]); 
      } else {
        expanded_coors.push_back(old_coors[i]);
      }
    }
  } else {
    exit(1);
  }

  return CalcEffOneDimArrayOffset(
      expanded_coors,
      blk_ndim,
      expanded_data_offsets
  );
}
} /* gqten */ 
