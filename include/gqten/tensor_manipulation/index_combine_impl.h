// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-06-28 17:12
* 
* Description: GraceQ/tensor project. Implementation details for Index combination.
*/
#include "gqten/gqten.h"

#include <utility>    // pair
#include <unordered_map>    // unordered_map

#include <assert.h>   // assert

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


// Forward declarations.
using QNSctIntraOffsetPair = std::pair<QNSector, size_t>;
using QNSctVec = std::vector<QNSector>;

class QNSctVecHashFunc {
public:
  size_t operator() (const QNSctVec &qnscts) const {
    return VecHasher(qnscts);  
  }
};

using QNSctVecQNSctIntraOffsetPairMap = std::unordered_map<
                                            QNSctVec,
                                            QNSctIntraOffsetPair,
                                            QNSctVecHashFunc
                                        >;
template <typename TenElemType>
using QNSctVecQNBlkPtr = std::unordered_map<
                             QNSctVec,
                             QNBlock<TenElemType> *,
                             QNSctVecHashFunc
                         >;


class QNHashFunc {
public:
  size_t operator() (const QN &qn) const {
    return qn.Hash();
  }
};

using QNDimMap = std::unordered_map<QN, long, QNHashFunc>;


/** Generate Index combiner.
 * The direction of the combined Index is OUT.
 */
template <typename TenElemType>
GQTensor<TenElemType> IndexCombine(
    const Index &idx1,                ///< The first Index to be combined.
    const Index &idx2,                ///< The second Index to be combined.
    const std::string &new_idx_dir    ///< The direction of the combined index.
) {
  // Calculate combined QNSectors information.
  auto idx1_dir = idx1.dir;
  auto idx2_dir = idx2.dir;
  QNSctVecQNSctIntraOffsetPairMap qnscts_qnsct_intra_offset_pair_map;
  QNDimMap qn_dim_map;
  for (auto &qnsct_from_idx1 : idx1.qnscts) {
    for (auto &qnsct_from_idx2 : idx2.qnscts) {
      QN combined_qn;
      if (idx1_dir == IN) {
        if (idx2_dir == IN) {
          combined_qn = qnsct_from_idx1.qn + qnsct_from_idx2.qn;  
        } else if (idx2_dir == OUT) {
          combined_qn = qnsct_from_idx1.qn - qnsct_from_idx2.qn;  
        } else {
          exit(1);
        }
      } else if (idx1_dir == OUT) {
        if (idx2_dir == IN) {
          combined_qn = qnsct_from_idx2.qn - qnsct_from_idx1.qn;  
        } else if (idx2_dir == OUT) {
          combined_qn = -qnsct_from_idx2.qn - qnsct_from_idx1.qn;  
        } else {
          exit(1);
        }
      } else {
        exit(1);
      }
      if (new_idx_dir == IN) {
        combined_qn = -combined_qn;
      }
      auto poss_it = qn_dim_map.find(combined_qn);
      long combined_dim = qnsct_from_idx1.dim * qnsct_from_idx2.dim;
      if (poss_it != qn_dim_map.cend()) {         // The combined QN exists.
        qnscts_qnsct_intra_offset_pair_map[
            {qnsct_from_idx1, qnsct_from_idx2}
        ] = std::make_pair(
            QNSector(combined_qn, 0),
            qn_dim_map[combined_qn]
        );
        qn_dim_map[combined_qn] += combined_dim;
      } else {                                    // The combined QN doesn't exist.
        qnscts_qnsct_intra_offset_pair_map[
            {qnsct_from_idx1, qnsct_from_idx2}
        ] = std::make_pair(QNSector(combined_qn, 0), 0);
        qn_dim_map[combined_qn] = combined_dim; 
      }
    }
  }
  // Reset combined_qnscts
  for (auto &kv : qnscts_qnsct_intra_offset_pair_map) {
    kv.second.first = QNSector(
                          kv.second.first.qn,
                          qn_dim_map[kv.second.first.qn]
                      );
  }
  
  // Generate QNBlocks of the combiner.
  QNSctVecQNBlkPtr<TenElemType> qnscts_pqnblk_map;
  for (auto kv : qnscts_qnsct_intra_offset_pair_map) {
    QNSctVec combiner_qnblk_qnscts = kv.first; 
    combiner_qnblk_qnscts.push_back(kv.second.first);
    auto poss_it = qnscts_pqnblk_map.find(combiner_qnblk_qnscts);
    if (poss_it != qnscts_pqnblk_map.cend()) {    // The QNBlock exists.
      auto pqnblk = poss_it->second;
      long combined_qnsct_coor = kv.second.second;
      for (long i = 0; i < combiner_qnblk_qnscts[0].dim; ++i) {
        for (long j = 0; j < combiner_qnblk_qnscts[1].dim; ++j) {
          (*pqnblk)({i, j, combined_qnsct_coor}) = 1.0;
          combined_qnsct_coor++;
        }
      }
    } else {                                      // The QNBlock doesn't exist.
      auto pqnblk = new QNBlock<TenElemType>(combiner_qnblk_qnscts); 
      long combined_qnsct_coor = kv.second.second;
      for (long i = 0; i < combiner_qnblk_qnscts[0].dim; ++i) {
        for (long j = 0; j < combiner_qnblk_qnscts[1].dim; ++j) {
          (*pqnblk)({i, j, combined_qnsct_coor}) = 1.0;
          combined_qnsct_coor++;
        }
      }
      qnscts_pqnblk_map[combiner_qnblk_qnscts] = pqnblk;
    }
  }

  // Generate combiner tensor.
  std::vector<QNSector> combined_qnscts;
  for (auto &kv : qn_dim_map) {
    combined_qnscts.push_back(QNSector(kv.first, kv.second));
  }
  Index combined_idx = Index(combined_qnscts, new_idx_dir);
  GQTensor<TenElemType> combiner({idx1, idx2, combined_idx});
  for (auto &kv : qnscts_pqnblk_map) {
    (combiner.blocks()).push_back(kv.second);
  }

  return combiner;
}
} /* gqten */ 
