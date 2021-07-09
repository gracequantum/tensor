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
#include "gqten/framework/hp_numeric/lapack.h"    // MatSVD

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


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


/**
Insert a list of data blocks. The BlockSparseDataTensor must be empty before
performing this insertion.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlksInsert(
    const std::vector<size_t> &blk_idxs,
    const std::vector<CoorsT> &blk_coors_s,
    const bool alloc_mem,
    const bool init
) {
  assert(blk_idx_data_blk_map_.empty());
  //it's better that every CoorsT is unique.
  auto iter = blk_idxs.begin();
  for(auto &blk_coors: blk_coors_s){
    size_t blk_idx = *iter;
    blk_idx_data_blk_map_[blk_idx] = DataBlk<QNT>(blk_coors, *pgqten_indexes);
    iter++;
  }
  raw_data_size_ = 0;
  for (auto &[idx, data_blk] : blk_idx_data_blk_map_) {
    data_blk.data_offset = raw_data_size_;
    raw_data_size_ += data_blk.size;
  }
  if (alloc_mem) {
    Allocate(init);
  } else {
    assert(pactual_raw_data_ == nullptr);
  }
}


/**
Insert a list of data blocks. The BlockSparseDataTensor must be empty before
performing this insertion.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlksInsert(
    const std::vector<CoorsT> &blk_coors_s,
    const bool alloc_mem,
    const bool init
) {
  assert(blk_idx_data_blk_map_.empty());
  //it's better that every CoorsT is unique.
  std::vector<size_t> blk_idxs;
  blk_idxs.reserve(blk_coors_s.size());
  for (auto &blk_coors: blk_coors_s) {
    blk_idxs.push_back(std::move(BlkCoorsToBlkIdx(blk_coors)));
  }
  DataBlksInsert(blk_idxs, blk_coors_s, alloc_mem, init);
}


/**
Copy and rescale raw data from another tensor.
*/
template<typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopyAndScale(
    const RawDataCopyAndScaleTask<ElemT> &task,
    const ElemT *pten_raw_data
) {
  RawDataCopyAndScale_(task, pten_raw_data);
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
  bool scalar_c_first_task = true;
#ifndef NDEBUG
  if (c_is_scalar) {
    assert(saved_axes_set[0].empty() && saved_axes_set[1].empty());
  }
#endif /* ifndef NDEBUG */
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
          GQTEN_Double beta;
          if (scalar_c_first_task) {
            beta = 0.0;
            raw_data_size_ = 1;     // Set raw data size at first task scheduling
            scalar_c_first_task = false;
          } else {
            beta = 1.0;
          }
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  a_b_need_trans.first,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  a_b_need_trans.second,
                  m, k, n,
                  beta
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
SVD decomposition.
*/
template <typename ElemT, typename QNT>
std::map<size_t, DataBlkMatSvdRes<ElemT>>
BlockSparseDataTensor<ElemT, QNT>::DataBlkDecompSVD(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map
) const {
  std::map<size_t, DataBlkMatSvdRes<ElemT>> idx_svd_res_map;
  for (auto &idx_data_blk_mat : idx_data_blk_mat_map) {
    auto idx = idx_data_blk_mat.first;
    auto data_blk_mat = idx_data_blk_mat.second;
    ElemT *mat = RawDataGenDenseDataBlkMat_(data_blk_mat);
    ElemT *u = nullptr;
    ElemT *vt = nullptr;
    GQTEN_Double *s = nullptr;
    size_t m = data_blk_mat.rows;
    size_t n = data_blk_mat.cols;
    size_t k = m > n ? n : m;
    hp_numeric::MatSVD(mat, m, n, u, s, vt);
    free(mat);
    idx_svd_res_map[idx] = DataBlkMatSvdRes<ElemT>(m, n, k, u, s, vt);
  }
  return idx_svd_res_map;
}


template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopySVDUdata(
    const CoorsT &blk_coors, const size_t mat_m, const size_t mat_n,
    const size_t row_offset,
    const ElemT *u, const size_t u_m, const size_t u_n,
    const std::vector<size_t> & kept_cols
) {
  assert(kept_cols.size() == mat_n);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  // TODO: Remove direct touch the raw data in DataBlk* member!
  auto data = pactual_raw_data_ + blk_idx_data_blk_map_[blk_idx].data_offset;
  size_t data_idx = 0;
  for (size_t i = 0; i < mat_m; ++i) {
    for (size_t j = 0; j < mat_n; ++j) {
      data[data_idx] = u[(row_offset + i) * u_n + kept_cols[j]];
      data_idx++;
    }
  }
}


template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkCopySVDVtData(
    const CoorsT &blk_coors, const size_t mat_m, const size_t mat_n,
    const size_t col_offset,
    const ElemT *vt, const size_t vt_m, const size_t vt_n,
    const std::vector<size_t> & kept_rows
) {
  assert(kept_rows.size() == mat_m);
  auto blk_idx = BlkCoorsToBlkIdx(blk_coors);
  // TODO: Remove direct touch the raw data in DataBlk* member!
  auto data = pactual_raw_data_ + blk_idx_data_blk_map_[blk_idx].data_offset;
  for (size_t i = 0; i < mat_m; ++i) {
    memcpy(
        data + (i * mat_n),
        vt + (kept_rows[i] * vt_n + col_offset),
        mat_n * sizeof(ElemT)
    );
  }
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
