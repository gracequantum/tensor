// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-26 21:30
*
* Description: GraceQ/tensor project. Global level operations in BlockSparseDataTensor.
*/

/**
@file global_operations.h
@brief Global level operations in BlockSparseDataTensor.
*/
#ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_GLOBAL_OPERATIONS_H
#define GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_GLOBAL_OPERATIONS_H


#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "gqten/gqtensor/blk_spar_data_ten/data_blk.h"                    // DataBlk
#include "gqten/gqtensor/blk_spar_data_ten/data_blk_operations.h"
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operations.h"
#include "gqten/framework/value_t.h"                                      // GQTEN_Double, GQTEN_Complex
#include "gqten/framework/hp_numeric/ten_trans.h"                         // TensorTranspose
#include "gqten/utility/utils_inl.h"                                      // CalcMultiDimDataOffsets, Reorder

#include <map>              // map
#include <unordered_set>    // unordered_set

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


/**
Clear all contents of this block sparse data tensor.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Clear(void) {
  DataBlkClear_();
  RawDataFree_();
}


/**
Allocate the memory based on the size of raw_data_size_;

@param init Whether initialize the memory to 0.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Allocate(const bool init) {
  RawDataAlloc_(raw_data_size_, init);
}


/**
Random set all elements in [0, 1].
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Random(void) {
  if (IsScalar()) { raw_data_size_ = 1; }
  if (raw_data_size_ > actual_raw_data_size_) {
    RawDataAlloc_(raw_data_size_);
  }
  RawDataRand_();
}


/**
Transpose the block sparse data tensor.

@param transed_idxes_order Transposed order of indexes.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Transpose(
    const std::vector<size_t> &transed_idxes_order
) {
  assert(transed_idxes_order.size() == blk_shape.size());
  // Give a shorted order, do nothing
  if (std::is_sorted(transed_idxes_order.begin(), transed_idxes_order.end())) {
    return;
  }

  Reorder(blk_shape, transed_idxes_order);
  blk_multi_dim_offsets_ = CalcMultiDimDataOffsets(blk_shape);

  std::vector<RawDataTransposeTask> raw_data_trans_tasks;
  BlkIdxDataBlkMap transed_blk_idx_data_blk_map;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_) {
    DataBlk<QNT> transed_data_blk(blk_idx_data_blk.second);
    transed_data_blk.Transpose(transed_idxes_order);
    auto transed_data_blk_idx = BlkCoorsToBlkIdx(transed_data_blk.blk_coors);
    transed_blk_idx_data_blk_map[transed_data_blk_idx] = transed_data_blk;
    raw_data_trans_tasks.push_back(
        RawDataTransposeTask(
            ten_rank,
            transed_idxes_order,
            blk_idx_data_blk.first,
            blk_idx_data_blk.second.shape,
            blk_idx_data_blk.second.data_offset,
            transed_data_blk_idx,
            transed_data_blk.shape
        )
    );
  }

  // Calculate and set data offset of each transposed data block.
  ResetDataOffset(transed_blk_idx_data_blk_map);
  RawDataTransposeTask::SortTasksByTranspoedBlkIdx(raw_data_trans_tasks);
  size_t trans_task_idx = 0;
  for (auto &blk_idx_data_blk : transed_blk_idx_data_blk_map) {
    raw_data_trans_tasks[trans_task_idx].transed_data_offset =
        blk_idx_data_blk.second.data_offset;
    trans_task_idx++;
  }
  // Update block index <-> data block map.
  blk_idx_data_blk_map_ = transed_blk_idx_data_blk_map;
  // Transpose the raw data.
  RawDataTransposeTask::SortTasksByOriginalBlkIdx(raw_data_trans_tasks);
  RawDataTranspose_(raw_data_trans_tasks);
}


/**
Normalize the data tensor and return its norm.

@return The norm before the normalization.
*/
template <typename ElemT, typename QNT>
GQTEN_Double BlockSparseDataTensor<ElemT, QNT>::Normalize(void) {
  return RawDataNormalize_();
}


/**
Complex conjugate.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Conj(void) {
  RawDataConj_();
}


/**
Add two input block sparse data tensor together and assign into this tensor.

@param a Block sparse data tensor A.
@param b Block sparse data tensor B.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::AddTwoBSDTAndAssignIn(
    const BlockSparseDataTensor &a,
    const BlockSparseDataTensor &b) {
  if (a.IsScalar() && b.IsScalar()) {
    ElemSet({}, a.ElemGet({}) + b.ElemGet({}));
    return;
  }

  auto blk_idx_data_blk_map_a = a.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_a;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_a) {
    auto data_blk = blk_idx_data_blk.second;
    DataBlkInsert(data_blk.blk_coors, false);
    raw_data_copy_tasks_a.push_back(
        RawDataCopyTask(data_blk.blk_coors, data_blk.data_offset, data_blk.size)
    );
  }

  auto blk_idx_data_blk_map_b = b.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_b;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_b) {
    auto blk_idx = blk_idx_data_blk.first;
    auto data_blk = blk_idx_data_blk.second;
    if (blk_idx_data_blk_map_a.find(blk_idx) != blk_idx_data_blk_map_a.end()) {
      raw_data_copy_tasks_b.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size,
              true
          )
      );
    } else {
      DataBlkInsert(data_blk.blk_coors, false);
      raw_data_copy_tasks_b.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size
          )
      );
    }
  }

  // Get data offset in destination.
  for (auto &task : raw_data_copy_tasks_a) {
    task.dest_data_offset = blk_idx_data_blk_map_[
                              BlkCoorsToBlkIdx(task.src_blk_coors)
                            ].data_offset;
  }
  for (auto &task : raw_data_copy_tasks_b) {
    task.dest_data_offset = blk_idx_data_blk_map_[
                                BlkCoorsToBlkIdx(task.src_blk_coors)
                            ].data_offset;
  }

  Allocate();
  RawDataCopy_(raw_data_copy_tasks_a, a.pactual_raw_data_);
  RawDataCopy_(raw_data_copy_tasks_b, b.pactual_raw_data_);
}


/**
Add another block sparse data tensor to this block sparse data tensor.

@param rhs Block sparse data tensor on the right hand side.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::AddAndAssignIn(
    const BlockSparseDataTensor &rhs) {
  assert(ten_rank == rhs.ten_rank);
  if (IsScalar() && rhs.IsScalar()) {
    ElemSet({}, ElemGet({}) + rhs.ElemGet({}));
    return;
  }

  // Copy block index <-> data block map and save actual raw data pointer.
  BlkIdxDataBlkMap this_blk_idx_data_blk_map(blk_idx_data_blk_map_);
  ElemT *this_pactual_raw_data_ = pactual_raw_data_;
  RawDataDiscard_();

  // Create raw data copy tasks for this tensor.
  std::vector<RawDataCopyTask> raw_data_copy_tasks_this;
  for (auto &blk_idx_data_blk : this_blk_idx_data_blk_map) {
    auto data_blk = blk_idx_data_blk.second;
    raw_data_copy_tasks_this.push_back(
        RawDataCopyTask(data_blk.blk_coors, data_blk.data_offset, data_blk.size)
    );
  }

  // Create raw data copy tasks for tensor on the right hand side.
  auto blk_idx_data_blk_map_rhs = rhs.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_rhs;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_rhs) {
    auto blk_idx = blk_idx_data_blk.first;
    auto data_blk = blk_idx_data_blk.second;
    if (blk_idx_data_blk_map_.find(blk_idx) != blk_idx_data_blk_map_.end()) {
      raw_data_copy_tasks_rhs.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size,
              true
          )
      );
    } else {
      DataBlkInsert(data_blk.blk_coors, false);
      raw_data_copy_tasks_rhs.push_back(
          RawDataCopyTask(
              data_blk.blk_coors,
              data_blk.data_offset,
              data_blk.size
          )
      );
    }
  }

  // Get data offset in result block sparse data tensor.
  for (auto &task : raw_data_copy_tasks_this) {
    task.dest_data_offset = blk_idx_data_blk_map_[
                              BlkCoorsToBlkIdx(task.src_blk_coors)
                            ].data_offset;
  }
  for (auto &task : raw_data_copy_tasks_rhs) {
    task.dest_data_offset = blk_idx_data_blk_map_[
                                BlkCoorsToBlkIdx(task.src_blk_coors)
                            ].data_offset;
  }

  Allocate();
  RawDataCopy_(raw_data_copy_tasks_this, this_pactual_raw_data_);
  free(this_pactual_raw_data_);
  RawDataCopy_(raw_data_copy_tasks_rhs, rhs.pactual_raw_data_);
}


/**
Multiply this block sparse data tensor by a scalar.

@param s A scalar.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::MultiplyByScalar(const ElemT s) {
  RawDataMultiplyByScalar_(s);
}


/**
Contract two block sparse data tensors follow a queue of raw data contraction
tasks.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::CtrctTwoBSDTAndAssignIn(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    std::vector<RawDataCtrctTask> &raw_data_ctrct_tasks
) {
  assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
  if (raw_data_ctrct_tasks.empty()) { return; }

  Allocate();

  bool a_need_trans = raw_data_ctrct_tasks[0].a_need_trans;
  bool b_need_trans = raw_data_ctrct_tasks[0].b_need_trans;
  std::unordered_map<size_t, ElemT *> a_blk_idx_transed_data_map;
  std::unordered_map<size_t, ElemT *> b_blk_idx_transed_data_map;
  RawDataCtrctTask::SortTasksByCBlkIdx(raw_data_ctrct_tasks);
  for (auto &task : raw_data_ctrct_tasks) {
    const ElemT *a_data;
    const ElemT *b_data;
    if (a_need_trans) {
      auto poss_it = a_blk_idx_transed_data_map.find(task.a_blk_idx);
      if (poss_it != a_blk_idx_transed_data_map.end()) {
        a_data = poss_it->second;
      } else {
        auto a_data_blk = bsdt_a.blk_idx_data_blk_map_.at(task.a_blk_idx);
        ElemT *transed_data = (ElemT *) malloc(a_data_blk.size * sizeof(ElemT));
        ShapeT a_blk_transed_shape(a_data_blk.shape);
        Reorder(a_blk_transed_shape, task.a_trans_orders);
        hp_numeric::TensorTranspose(
            task.a_trans_orders,
            bsdt_a.ten_rank,
            bsdt_a.pactual_raw_data_ + task.a_data_offset,
            a_data_blk.shape,
            transed_data,
            a_blk_transed_shape
        );
        a_blk_idx_transed_data_map[task.a_blk_idx] = transed_data;
        a_data = transed_data;
      }
    } else {
      a_data = bsdt_a.pactual_raw_data_ + task.a_data_offset;
    }
    if (b_need_trans) {
      auto poss_it = b_blk_idx_transed_data_map.find(task.b_blk_idx);
      if (poss_it != b_blk_idx_transed_data_map.end()) {
        b_data = poss_it->second;
      } else {
        auto b_data_blk = bsdt_b.blk_idx_data_blk_map_.at(task.b_blk_idx);
        ElemT *transed_data = (ElemT *) malloc(b_data_blk.size * sizeof(ElemT));
        ShapeT b_blk_transed_shape(b_data_blk.shape);
        Reorder(b_blk_transed_shape, task.b_trans_orders);
        hp_numeric::TensorTranspose(
            task.b_trans_orders,
            bsdt_b.ten_rank,
            bsdt_b.pactual_raw_data_ + task.b_data_offset,
            b_data_blk.shape,
            transed_data,
            b_blk_transed_shape
        );
        b_blk_idx_transed_data_map[task.b_blk_idx] = transed_data;
        b_data = transed_data;
      }
    } else {
      b_data = bsdt_b.pactual_raw_data_ + task.b_data_offset;
    }
    RawDataTwoMatMultiplyAndAssignIn_(
        a_data,
        b_data,
        task.c_data_offset,
        task.m, task.k, task.n,
        task.beta
    );
  }

  for (auto &blk_idx_transed_data : a_blk_idx_transed_data_map) {
    free(blk_idx_transed_data.second);
  }
  for (auto &blk_idx_transed_data : b_blk_idx_transed_data_map) {
    free(blk_idx_transed_data.second);
  }
}


// Helpers for tensor expansion
using BlkCoorsShapePair = std::pair<CoorsT, ShapeT>;
// (hash value of qn info) -> (blk coors, shape)
using QnInfoHashBlkCoorsShapeMap = std::unordered_map<
                                       size_t,
                                       BlkCoorsShapePair
                                   >;


template <typename QNT>
inline size_t CalcDataBlkResidueDimSize(const DataBlk<QNT> &data_blk) {
  return data_blk.size / data_blk.shape[0];
}


/**
Construct tensor expansion data over the first index, from corresponding BSDTs.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ConstructExpandedDataOnFirstIndex(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    const std::vector<bool> &is_a_first_idx_qnsct_expanded,
    const std::vector<bool> &is_b_first_idx_qnsct_expanded,
    const std::map<size_t, size_t> &b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
) {
  // Create data blocks and copy data/fill zeros tasks
  auto blk_idx_data_blk_map_a = bsdt_a.GetBlkIdxDataBlkMap();
  std::unordered_set<size_t> created_leading_dims_by_bsdt_a;
  std::vector<RawDataCopyTask> raw_data_copy_tasks_a;
  std::vector<RawDataSetZerosTask> raw_data_set_zeros_tasks_a;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_a) {
    auto data_blk = blk_idx_data_blk.second;
    auto blk_coors = data_blk.blk_coors;
    auto pblk_idx_data_blk = DataBlkInsert(blk_coors, false);
    created_leading_dims_by_bsdt_a.insert(blk_coors[0]);
    // Create copy task
    raw_data_copy_tasks_a.push_back(
        // Block coordinates in the extended tensor always equals to block coordinates in the A tensor
        RawDataCopyTask(blk_coors, data_blk.data_offset, data_blk.size)
    );
    if (is_a_first_idx_qnsct_expanded[blk_coors[0]]) {
      auto set_zeros_data_size =
           CalcDataBlkResidueDimSize(data_blk) *
           (pblk_idx_data_blk->second.shape[0] - data_blk.shape[0]);
      raw_data_set_zeros_tasks_a.push_back(
          RawDataSetZerosTask(blk_coors, set_zeros_data_size, data_blk.size)
      );
    }
  }
  auto blk_idx_data_blk_map_b = bsdt_b.GetBlkIdxDataBlkMap();
  std::vector<RawDataCopyTask> raw_data_copy_tasks_b;
  std::vector<RawDataSetZerosTask> raw_data_set_zeros_tasks_b;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_b) {
    auto data_blk = blk_idx_data_blk.second;
    auto blk_coors = data_blk.blk_coors;
    auto expanded_leading_dim_blk_coor =
         b_idx_qnsct_coor_expanded_idx_qnsct_coor_map.at(blk_coors[0]);
    auto expanded_blk_coors = blk_coors;
    expanded_blk_coors[0] = expanded_leading_dim_blk_coor;
    // Data block has not been created, insert the data block
    // If the leading dimension expanded, fill zeros
    if (
        created_leading_dims_by_bsdt_a.find(expanded_leading_dim_blk_coor) ==
        created_leading_dims_by_bsdt_a.end()
    ) {
      auto pblk_idx_data_blk = DataBlkInsert(expanded_blk_coors, false);
      if (is_b_first_idx_qnsct_expanded[blk_coors[0]]) {
        auto set_zeros_data_size =
             CalcDataBlkResidueDimSize(data_blk) *
             (pblk_idx_data_blk->second.shape[0] - data_blk.shape[0]);
        raw_data_set_zeros_tasks_b.push_back(
            RawDataSetZerosTask(expanded_blk_coors, set_zeros_data_size, 0)
        );
      }
    } else {    // Remove fill zeros task created by walking through BSDTa
      for (
          auto it = raw_data_set_zeros_tasks_a.begin();
          it != raw_data_set_zeros_tasks_a.end();
          ++it
      ) {
        if (it->blk_coors == expanded_blk_coors) {
          raw_data_set_zeros_tasks_a.erase(it);
          break;
        }
      }
    }
    // Create copy task
    RawDataCopyTask copy_task(blk_coors, data_blk.data_offset, data_blk.size);
    copy_task.dest_blk_coors = expanded_blk_coors;
    raw_data_copy_tasks_b.push_back(copy_task);
  }

  // Copy task: set data offset in destination
  for (auto &copy_task : raw_data_copy_tasks_a) {
    copy_task.dest_data_offset = blk_idx_data_blk_map_.at(
                                     BlkCoorsToBlkIdx(copy_task.src_blk_coors)
                                 ).data_offset;
  }
  for (auto &copy_task : raw_data_copy_tasks_b) {
    auto dest_data_blk = blk_idx_data_blk_map_.at(
                                BlkCoorsToBlkIdx(copy_task.dest_blk_coors)
                         );
    auto dest_data_offset = dest_data_blk.data_offset;
    // If block expanded, extra data offset needed
    if (is_b_first_idx_qnsct_expanded[copy_task.src_blk_coors[0]]) {
      auto src_data_blk = bsdt_b.blk_idx_data_blk_map_.at(
                              BlkCoorsToBlkIdx(copy_task.src_blk_coors)
                          );
      size_t extra_data_offset =
             CalcDataBlkResidueDimSize(dest_data_blk) *
             (dest_data_blk.shape[0] - src_data_blk.shape[0]);
      dest_data_offset += extra_data_offset;
    }
    copy_task.dest_data_offset = dest_data_offset;
  }
  // Set zeros task: set data offset
  for (auto &task : raw_data_set_zeros_tasks_a) {
    task.data_offset = blk_idx_data_blk_map_.at(
                           BlkCoorsToBlkIdx(task.blk_coors)
                       ).data_offset + task.extra_data_offset;
  }
  for (auto &task : raw_data_set_zeros_tasks_b) {
    task.data_offset = blk_idx_data_blk_map_.at(
                           BlkCoorsToBlkIdx(task.blk_coors)
                       ).data_offset + task.extra_data_offset;
  }

  // Allocate memory
  Allocate();
  // Do data copy
  RawDataCopy_(raw_data_copy_tasks_a, bsdt_a.pactual_raw_data_);
  RawDataCopy_(raw_data_copy_tasks_b, bsdt_b.pactual_raw_data_);
  // Do zeros fill
  RawDataSetZeros_(raw_data_set_zeros_tasks_a);
  RawDataSetZeros_(raw_data_set_zeros_tasks_b);
}


/**
Copy contents from a real block sparse data tensor.

@param real_bsdt A real block sparse data tensor.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::CopyFromReal(
    const BlockSparseDataTensor<GQTEN_Double, QNT> &real_bsdt
) {
  Clear();
  if (std::is_same<ElemT, GQTEN_Complex>::value) {
    for (auto &blk_idx_data_blk : real_bsdt.GetBlkIdxDataBlkMap()) {
      DataBlkInsert(blk_idx_data_blk.second.blk_coors, false);
    }
    if (IsScalar() && (real_bsdt.GetActualRawDataSize() != 0)) {
      raw_data_size_ = 1;
    }

    Allocate();
    RawDataDuplicateFromReal_(
        real_bsdt.GetActualRawDataPtr(),
        real_bsdt.GetActualRawDataSize()
    );
  } else {
    assert(false);    // TODO: To-be implemented!
  }
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_GLOBAL_OPERATIONS_H */
