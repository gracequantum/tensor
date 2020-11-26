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
#include "gqten/framework/value_t.h"                                      // GQTEN_Double, GQTEN_Complex
#include "gqten/gqtensor/blk_spar_data_ten/data_blk.h"                    // DataBlk
#include "gqten/utility/utils_inl.h"                                      // CalcMultiDimDataOffsets, Reorder

#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


/**
Clear data blocks and reset raw_data_size_.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkClear_(void) {
  blk_idx_data_blk_map_.clear();
  raw_data_size_ = 0;
}


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
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Allocate(void) {
  RawDataAlloc_(raw_data_size_);
}


/**
Random set all elements in [0, 1].
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Random(void) {
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
