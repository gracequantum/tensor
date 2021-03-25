// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-03-23 10:00
*
* Description: GraceQ/tensor project. Expand two tensors.
*/

/**
@file ten_expand.h
@brief Expand two tensors.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_EXPAND_H
#define GQTEN_TENSOR_MANIPULATION_TEN_EXPAND_H


#include "gqten/gqtensor_all.h"     // GQTensor

#include <vector>       // vector
#include <algorithm>    // find

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


// Forward declarations
template <typename QNT>
class QNSectorHashFunc {
public:
  size_t operator() (const QNSector<QNT> &qnsct) const {
    return qnsct.Hash();
  }
};

template <typename QNT>
using QNSectorExpandMap = std::unordered_map<QNSector<QNT>, QNSector<QNT>, QNSectorHashFunc<QNT>>;

template <typename QNT>
using QNSectorExpandMapVec = std::vector<QNSectorExpandMap<QNT>>;

/**
The direction of the expansion.
*/
enum ExpandDirection {
  Up,     ///< Expand to up.
  Down    ///< Expand to down.
};

template <typename QNT>
IndexVec<QNT> ExpandIdxs(
    const IndexVec<QNT> &,
    const IndexVec<QNT> &,
    const std::vector<size_t> &,
    QNSectorExpandMapVec<QNT> &,
    QNSectorExpandMapVec<QNT> &
);

template <typename QNT>
inline Index<QNT> ExpandIndex(
    const Index<QNT> &,
    const Index<QNT> &,
    QNSectorExpandMapVec<QNT> &,
    QNSectorExpandMapVec<QNT> &
);


// Inline helpers for tensor expansion
template <typename TenElemT, typename QNT>
inline void ExpandedTenDivChecker(
    const GQTensor<TenElemT, QNT> &a,
    const GQTensor<TenElemT, QNT> &b,
    const GQTensor<TenElemT, QNT> &c
) {
  if (a.Div() != QNT()) {
    assert(c.Div() == a.Div());
  } else if (b.Div() != QNT()) {
    assert(c.Div() == b.Div());
  } else {
    assert(c.Div() == QNT());
  }
}


/**
Function version for tensor expansion.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param expand_idx_nums Index numbers which index to be expanded.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
void Expand(
    const GQTensor<TenElemT, QNT> *pa,
    const GQTensor<TenElemT, QNT> *pb,
    const std::vector<size_t> &expand_idx_nums,
    GQTensor<TenElemT, QNT> *pc
) {
#ifndef NDEBUG
  assert(pa->Rank() == pb->Rank());     // To be expanded tensors should have the same rank
  for (size_t i = 0; i < pa->Rank(); ++i) {
    // Indexes of the to be expanded tensors should have the same directions
    assert(pa->GetIndexes()[i].GetDir() == pb->GetIndexes()[i].GetDir());
  }
  // To be expanded tensors should have the same quantum number divergence or a null quantum number divergence QNT()
  assert(pa->Div() ==  pb->Div() || pa->Div() == QNT() || pb->Div() == QNT());
#endif /* ifndef NDEBUG */

  // Expand indexes
  QNSectorExpandMapVec<QNT> a_idx_qnsct_expand_maps, b_idx_qnsct_expand_maps;
  auto expanded_idxs = ExpandIdxs(
                           pa->GetIndexes(),
                           pb->GetIndexes(),
                           expand_idx_nums,
                           a_idx_qnsct_expand_maps,
                           b_idx_qnsct_expand_maps
                       );

  // Expand data
  (*pc) = GQTensor<TenElemT, QNT>(expanded_idxs);
  (pc->GetBlkSparDataTen()).ConstructExpandedDataFrom(
      pa->GetBlkSparDataTen(),
      pb->GetBlkSparDataTen()
  );

#ifndef NDEBUG
  ExpandedTenDivChecker(*pa, *pb, *pc);
#endif /* ifndef NDEBUG */
}


/**
Expand Index of tensor \f$ A \f$ and \f$ B \f$.

@tparam QNT The quantum number type of the tensors.

@param a_idxs Indexes of tensor \f$ A \f$.
@param b_idxs Indexes of tensor \f$ B \f$.
@param expand_idx_nums Index numbers which index to be expanded.
@param a_idx_qnsct_expand_maps Mappings from QNSectors of Index of A to corresponding expanded QNSectors of Index of C.
@param b_idx_qnsct_expand_maps Mappings from QNSectors of Index of B to corresponding expanded QNSectors of Index of C.
*/
template <typename QNT>
IndexVec<QNT> ExpandIdxs(
    const IndexVec<QNT> &a_idxs,
    const IndexVec<QNT> &b_idxs,
    const std::vector<size_t> &expand_idx_nums,
    QNSectorExpandMapVec<QNT> &a_idx_qnsct_expand_maps,
    QNSectorExpandMapVec<QNT> &b_idx_qnsct_expand_maps
) {
  IndexVec<QNT> expanded_idxs;
  size_t ten_rank = a_idxs.size();
  for (size_t i = 0; i < ten_rank; ++i) {
    auto poss_it = std::find(expand_idx_nums.begin(), expand_idx_nums.end(), i);
    if (poss_it != expand_idx_nums.end()) {     // Corresponding Index needs to expand
      expanded_idxs.push_back(
          ExpandIndex(
              a_idxs[i],
              b_idxs[i],
              a_idx_qnsct_expand_maps,
              b_idx_qnsct_expand_maps
          )
      );
    } else {                                    // Corresponding Index does not need to expand
      expanded_idxs.push_back(a_idxs[i]);
    }
  }

  return expanded_idxs;
}


template <typename QNT>
inline Index<QNT> ExpandIndex(
    const Index<QNT> &idx_from_a,
    const Index<QNT> &idx_from_b,
    QNSectorExpandMapVec<QNT> &a_idx_qnsct_expand_maps,
    QNSectorExpandMapVec<QNT> &b_idx_qnsct_expand_maps
) {
  QNSectorVec<QNT> expanded_qnscts;
  QNSectorExpandMap<QNT> a_idx_qnsct_expand_map, b_idx_qnsct_expand_map;
  QNSectorVec<QNT> qnscts_from_b;
  for (
      size_t sct_coor_b = 0; sct_coor_b < idx_from_b.GetQNSctNum(); ++sct_coor_b
  ) {
    qnscts_from_b.push_back(idx_from_b.GetQNSct(sct_coor_b));
  }
  for (
      size_t sct_coor_a = 0; sct_coor_a < idx_from_a.GetQNSctNum(); ++sct_coor_a
  ) {
    auto qnsct_from_a = idx_from_a.GetQNSct(sct_coor_a);
    auto qnscts_from_b_size = qnscts_from_b.size();
    bool has_matched_qnsct_in_qnscts_from_b = false;
    for (size_t sct_coor_b = 0; sct_coor_b < qnscts_from_b_size; ++sct_coor_b) {
      auto qnsct_from_b = qnscts_from_b[sct_coor_b];
      if (qnsct_from_b == qnsct_from_a) {
        auto expanded_qnsct = QNSector<QNT>(
                                  qnsct_from_a.GetQn(),
                                  qnsct_from_a.dim() + qnsct_from_b.dim()
                              );
        expanded_qnscts.push_back(expanded_qnsct);
        a_idx_qnsct_expand_map[qnsct_from_a] = expanded_qnsct;
        b_idx_qnsct_expand_map[qnsct_from_b] = expanded_qnsct;
        qnscts_from_b.erase(qnscts_from_b.begin() + sct_coor_b);
        has_matched_qnsct_in_qnscts_from_b = true;
        break;
      }
    }
    if (!has_matched_qnsct_in_qnscts_from_b) {
      expanded_qnscts.push_back(qnsct_from_a);
      a_idx_qnsct_expand_map[qnsct_from_a] = qnsct_from_a;
    }
  }

  // Deal with left QNSectors from the index from tensor B
  for (auto &qnsct_from_b : qnscts_from_b) {
    expanded_qnscts.push_back(qnsct_from_b);
    b_idx_qnsct_expand_map[qnsct_from_b] = qnsct_from_b;
  }

  a_idx_qnsct_expand_maps.push_back(a_idx_qnsct_expand_map);
  b_idx_qnsct_expand_maps.push_back(b_idx_qnsct_expand_map);
  return Index<QNT>(expanded_qnscts, idx_from_a.GetDir());
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_EXPAND_H */
