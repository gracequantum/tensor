// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-30 11:57
*
* Description: GraceQ/tensor project. Data block level operations for block
* sparse data tensor.
*/

/**
@file data_blk_operations.h
@brief Data block level operations for block sparse data tensor.
*/
#ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_LEVEL_OPERATIONS_H
#define GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_LEVEL_OPERATIONS_H


#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operations.h"
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operation_tasks.h"

#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


/**
Insert a new data block. User can decide whether allocate the memory.

@param blk_coors Block coordinates of the new data block.
@param alloc_mem Whether allocate the memory and set them to 0.

@return An iterator for this new block index <-> data block pair.

@note You can only gap the memory allocation procedure when the raw data is empty.
*/
template <typename ElemT, typename QNT>
typename BlockSparseDataTensor<ElemT, QNT>::BlkIdxDataBlkMap::iterator
BlockSparseDataTensor<ElemT, QNT>::DataBlkInsert(
    const CoorsT &blk_coors, const bool alloc_mem
) {
  assert(!blk_coors.empty());
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  blk_idx_data_blk_map_[blk_idx] = DataBlk<QNT>(blk_coors, *pgqten_indexes);
  size_t inserted_data_size = blk_idx_data_blk_map_[blk_idx].size;
  size_t total_data_offset = 0;
  for (auto &idx_blk : blk_idx_data_blk_map_) {
    if (idx_blk.first < blk_idx) {
      total_data_offset += idx_blk.second.size;
    } else if (idx_blk.first > blk_idx) {
      idx_blk.second.data_offset += inserted_data_size;
    } else {
      idx_blk.second.data_offset = total_data_offset;
    }
  }
  raw_data_size_ += inserted_data_size;

  if (alloc_mem) {
    RawDataInsert_(total_data_offset, inserted_data_size, true);
  } else {
    assert(pactual_raw_data_ == nullptr);
  }

  return blk_idx_data_blk_map_.find(blk_idx);
}


// Some helpers for tensor contraction
std::vector<std::vector<size_t>> TenCtrctGenSavedAxesSet(
    const size_t a_rank,
    const size_t b_rank,
    const std::vector<std::vector<size_t>> &ctrct_axes_set
) {
  auto a_ctrct_axes = ctrct_axes_set[0];
  auto b_ctrct_axes = ctrct_axes_set[1];
  std::vector<std::vector<size_t>> saved_axes_set;
  saved_axes_set.reserve(2);
  std::vector<size_t> a_saved_axes;
  for (size_t i = 0; i < a_rank; ++i) {
    if (std::find(a_ctrct_axes.begin(), a_ctrct_axes.end(), i) ==
        a_ctrct_axes.end()
    ) {
      a_saved_axes.push_back(i);
    }
  }
  saved_axes_set.emplace_back(a_saved_axes);
  std::vector<size_t> b_saved_axes;
  for (size_t i = 0; i < b_rank; ++i) {
    if (std::find(b_ctrct_axes.begin(), b_ctrct_axes.end(), i) ==
        b_ctrct_axes.end()
    ) {
      b_saved_axes.push_back(i);
    }
  }
  saved_axes_set.emplace_back(b_saved_axes);
  return saved_axes_set;
}


std::pair<bool, bool> TenCtrctNeedTransCheck(
    const std::vector<std::vector<size_t>> &ctrct_axes_set,
    const std::vector<std::vector<size_t>> &saved_axes_set,
    std::vector<size_t> &a_trans_orders,
    std::vector<size_t> &b_trans_orders
) {
  a_trans_orders = saved_axes_set[0];
  a_trans_orders.insert(
      a_trans_orders.end(),
      ctrct_axes_set[0].begin(),
      ctrct_axes_set[0].end()
  );
  bool a_need_trans;
  if (std::is_sorted(a_trans_orders.begin(), a_trans_orders.end())) {
    a_need_trans = false;
  } else {
    a_need_trans = true;
  }

  b_trans_orders = ctrct_axes_set[1];
  b_trans_orders.insert(
      b_trans_orders.end(),
      saved_axes_set[1].begin(),
      saved_axes_set[1].end()
  );
  bool b_need_trans;
  if (std::is_sorted(b_trans_orders.begin(), b_trans_orders.end())) {
    b_need_trans = false;
  } else {
    b_need_trans = true;
  }

  return std::make_pair(a_need_trans, b_need_trans);
}


CoorsT GenTenCtrctDataBlkCoors(
    const CoorsT &a_blk_coors,
    const CoorsT &b_blk_coors,
    const std::vector<std::vector<size_t>> &saved_axes_set
) {
  CoorsT c_blk_coors;
  for (auto axis : saved_axes_set[0]) {
    c_blk_coors.push_back(a_blk_coors[axis]);
  }
  for (auto axis : saved_axes_set[1]) {
    c_blk_coors.push_back(b_blk_coors[axis]);
  }
  return c_blk_coors;
}


/**
Generate data blocks for two tensor contraction.

@param bsdt_a Block sparse data tensor A.
@param bsdt_b Block sparse data tensor B.
@param ctrct_axes_set To-be contracted tensor axes indexes.
       For example, {{0, 1}, {3, 2}}.
*/
template <typename ElemT, typename QNT>
std::vector<RawDataCtrctTask>
BlockSparseDataTensor<ElemT, QNT>::DataBlkGenForTenCtrct(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    const std::vector<std::vector<size_t>> &ctrct_axes_set
) {
  assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
  auto saved_axes_set = TenCtrctGenSavedAxesSet(
                            bsdt_a.ten_rank,
                            bsdt_b.ten_rank,
                            ctrct_axes_set
                        );
  std::vector<size_t> a_trans_orders, b_trans_orders;
  auto a_b_need_trans = TenCtrctNeedTransCheck(
                            ctrct_axes_set,
                            saved_axes_set,
                            a_trans_orders,
                            b_trans_orders
                        );
  auto a_blk_idx_data_blk_map = bsdt_a.GetBlkIdxDataBlkMap();
  auto b_blk_idx_data_blk_map = bsdt_b.GetBlkIdxDataBlkMap();
  auto a_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkInfoPartHashMap(
                                                a_blk_idx_data_blk_map,
                                                ctrct_axes_set[0]
                                            );
  auto b_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkInfoPartHashMap(
                                                b_blk_idx_data_blk_map,
                                                ctrct_axes_set[1]
                                            );
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks;
  std::unordered_map<size_t, size_t>
      a_blk_idx_m_map,
      a_blk_idx_k_map,
      b_blk_idx_n_map;

  bool c_is_scalar = IsScalar();
  if (c_is_scalar) {
    assert(saved_axes_set[0].empty() && saved_axes_set[1].empty());
    raw_data_size_ = 1;
  }
  for (auto &a_blk_idx_part_hash : a_blk_idx_qnblk_info_part_hash_map) {
    for (auto &b_blk_idx_part_hash : b_blk_idx_qnblk_info_part_hash_map) {
      if (a_blk_idx_part_hash.second == b_blk_idx_part_hash.second) {
        auto a_blk_idx = a_blk_idx_part_hash.first;
        auto b_blk_idx = b_blk_idx_part_hash.first;
        auto a_data_blk = a_blk_idx_data_blk_map[a_blk_idx];
        auto b_data_blk = b_blk_idx_data_blk_map[b_blk_idx];
        // Calculate m, k, n
        size_t m, k, n;
        if (c_is_scalar) {
          m = 1;
          n = 1;
        } else {
          if (a_blk_idx_m_map.find(a_blk_idx) != a_blk_idx_m_map.end()) {
            m = a_blk_idx_m_map.at(a_blk_idx);
          } else {
            m = VecMultiSelectElemts(a_data_blk.shape, saved_axes_set[0]);
            a_blk_idx_m_map[a_blk_idx] = m;
          }
          if (b_blk_idx_n_map.find(b_blk_idx) != b_blk_idx_n_map.end()) {
            n = b_blk_idx_n_map.at(b_blk_idx);
          } else {
            n = VecMultiSelectElemts(b_data_blk.shape, saved_axes_set[1]);
            b_blk_idx_n_map[b_blk_idx] = n;
          }
        }
        if (a_blk_idx_k_map.find(a_blk_idx) != a_blk_idx_k_map.end()) {
          k = a_blk_idx_k_map.at(a_blk_idx);
        } else {
          k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          a_blk_idx_k_map[a_blk_idx] = k;
        }

        // Create raw data contraction task
        if (c_is_scalar) {
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  a_b_need_trans.first,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  a_b_need_trans.second,
                  m, k, n,
                  1.0
              )
          );
        } else {
          auto c_blk_coors = GenTenCtrctDataBlkCoors(
                                 a_data_blk.blk_coors,
                                 b_data_blk.blk_coors,
                                 saved_axes_set
                             );
          auto c_blk_idx = BlkCoorsToBlkIdx(c_blk_coors);
          GQTEN_Double beta;
          if (blk_idx_data_blk_map_.find(c_blk_idx) !=
              blk_idx_data_blk_map_.end()
          ) {
            beta = 1.0;
          } else {
            DataBlkInsert(c_blk_coors, false);
            beta = 0.0;
          }
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  a_b_need_trans.first,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  a_b_need_trans.second,
                  c_blk_idx,
                  m, k, n,
                  beta
              )
          );
        }
      }
    }
  }

  for (auto &task : raw_data_ctrct_tasks) {
    if (a_b_need_trans.first) {
      task.a_trans_orders = a_trans_orders;
    }
    if (a_b_need_trans.second) {
      task.b_trans_orders = b_trans_orders;
    }
    if (!c_is_scalar) {
      task.c_data_offset = blk_idx_data_blk_map_[task.c_blk_idx].data_offset;
    } else {
      task.c_data_offset = 0;
    }
  }

  return raw_data_ctrct_tasks;
}


/**
Clear data blocks and reset raw_data_size_.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkClear_(void) {
  blk_idx_data_blk_map_.clear();
  raw_data_size_ = 0;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_DATA_BLK_LEVEL_OPERATIONS_H */