// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-15 12:17
*
* Description: GraceQ/tensor project. Block sparse data tensor.
*/

/**
@file blk_spar_data_ten.h
@brief Block sparse data tensor.
*/
#ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H
#define GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H


#include "gqten/framework/value_t.h"                                      // CoorsT, ShapeT
#include "gqten/framework/bases/streamable.h"                             // Streamable
#include "gqten/framework/hp_numeric/ten_trans.h"                         // TensorTranspose
#include "gqten/framework/hp_numeric/blas_level1.h"                       // VectorAddTo
#include "gqten/gqtensor/index.h"                                         // IndexVec, CalcQNSctNumOfIdxs
#include "gqten/gqtensor/blk_spar_data_ten/data_blk.h"                    // DataBlk
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operation_tasks.h"    // RawDataTransposeTask
#include "gqten/utility/utils_inl.h"                                      // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets, Rand, Reorder, CalcScalarNorm2, CalcConj

#include <map>          // map
#include <cmath>        // sqrt
#include <iostream>     // endl, istream, ostream

#include <stdlib.h>     // malloc
#include <string.h>     // memcpy, memset
#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


// Some helpers


/**
Block sparse data tensor.

@tparam ElemT Type of the tensor element.
@tparam QNT   Type of the quantum number.
*/
template <typename ElemT, typename QNT>
class BlockSparseDataTensor : public Streamable {
public:
  /// Type of block index to data block ordered mapping.
  using BlkIdxDataBlkMap = std::map<size_t, DataBlk<QNT>>;

  // Constructors and destructor.
  BlockSparseDataTensor(const IndexVec<QNT> *);
  BlockSparseDataTensor(const BlockSparseDataTensor &);
  BlockSparseDataTensor &operator=(const BlockSparseDataTensor &);
  ~BlockSparseDataTensor(void);

  // Element level operations.
  ElemT ElemGet(const std::pair<CoorsT, CoorsT> &) const;

  void ElemSet(const std::pair<CoorsT, CoorsT> &, const ElemT);

  // Data block level operations
  typename BlkIdxDataBlkMap::iterator
  DataBlkInsert(const CoorsT &blk_coors, const bool alloc_mem = true);

  // Global level operations
  void Clear(void);
  void Allocate(void);
  void Random(void);
  void Transpose(const std::vector<size_t> &);
  GQTEN_Double Normalize(void);
  void Conj(void);
  void AddTwoBSDTAndAssignIn(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &
  );
  void AddAndAssignIn(const BlockSparseDataTensor &);
  void MultiplyByScalar(const ElemT);
  void CopyFromReal(const BlockSparseDataTensor<GQTEN_Double, QNT> &);

  // Operators overload
  bool operator==(const BlockSparseDataTensor &) const;
  bool operator!=(const BlockSparseDataTensor &rhs) const {
    return !(*this == rhs);
  }

  // Override base class
  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

  // Misc
  /**
  Calculate block index from block coordinates.

  @param blk_coors Block coordinates.

  @return Corresponding block index.
  */
  size_t BlkCoorsToBlkIdx(const CoorsT &blk_coors) const {
    return CalcEffOneDimArrayOffset(blk_coors, blk_multi_dim_offsets);
  }

  /// Get block index <-> data block map.
  const BlkIdxDataBlkMap &GetBlkIdxDataBlkMap(void) const {
    return blk_idx_data_blk_map_;
  }

  /// Get the pointer to actual raw data constant.
  const ElemT *GetActualRawDataPtr(void) const { return pactual_raw_data_; }

  /// Get the actual raw data size.
  size_t GetActualRawDataSize(void) const { return actual_raw_data_size_; }

  // Static members.
  static void ResetDataOffset(BlkIdxDataBlkMap &);


  /// Rank of the tensor.
  size_t ten_rank = 0;
  /// Block shape.
  ShapeT blk_shape;
  /// Block multi-dimension data offsets;
  std::vector<size_t> blk_multi_dim_offsets;
  /// A pointer which point to the indexes of corresponding GQTensor.
  const IndexVec<QNT> *pgqten_indexes = nullptr;

private:
  /**
  Pointer which point to the actual one-dimensional data array (the raw data).
  @note This variable will only be changed in Constructors and RawData* member functions.
  */
  ElemT *pactual_raw_data_ = nullptr;

  /**
  Actual size of the raw data. This size must equal to the of pactual_raw_data_.
  @note This variable will only be changed in Constructors and RawData* member functions.
  */
  size_t actual_raw_data_size_ = 0;

  /**
  Size of the raw data in this block sparse data tensor. This size must equal to
  the sum of the size of each existed DataBlk.
  @note This variable will only be changed in DataBlk* member functions.
  */
  size_t raw_data_size_ = 0;

  /// Ordered map from block index to data block for existed blocks.
  BlkIdxDataBlkMap blk_idx_data_blk_map_;

  // Private data block operations.
  void DataBlkClear_(void);

  // Raw data operations.
  void RawDataFree_(void);
  void RawDataDiscard_(void);

  void RawDataAlloc_(const size_t);
  void RawDataInsert_(const size_t, const size_t, const bool init = false);

  void RawDataCopy_(const std::vector<RawDataCopyTask> &, const ElemT *);
  void RawDataDuplicateFromReal_(const GQTEN_Double *, const size_t);

  void RawDataRand_(void);
  void RawDataTranspose_(const std::vector<RawDataTransposeTask> &);
  GQTEN_Double RawDataNormalize_(void);
  void RawDataConj_(void);

  void RawDataMultiplyByScalar_(const ElemT);

  void RawDataRead_(std::istream &);
  void RawDataWrite_(std::ostream &) const;
};


/**
Create a block sparse data tensor using a pointer which point to the indexes
of corresponding GQTensor.

@param pgqten_indexes A pointer which point to the indexes of corresponding
       GQTensor.
*/
template <typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT>::BlockSparseDataTensor(
    const IndexVec<QNT> *pgqten_indexes
) : pgqten_indexes(pgqten_indexes) {
  ten_rank = pgqten_indexes->size();
  blk_shape = CalcQNSctNumOfIdxs(*pgqten_indexes);
  blk_multi_dim_offsets = CalcMultiDimDataOffsets(blk_shape);
}


/**
Copy a block sparse data tensor.

@param bsdt Another block sparse data tensor.
*/
template <typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT>::BlockSparseDataTensor(
    const BlockSparseDataTensor &bsdt
) :
    ten_rank(bsdt.ten_rank),
    blk_shape(bsdt.blk_shape),
    blk_multi_dim_offsets(bsdt.blk_multi_dim_offsets),
    blk_idx_data_blk_map_(bsdt.blk_idx_data_blk_map_),
    pgqten_indexes(bsdt.pgqten_indexes),
    raw_data_size_(bsdt.raw_data_size_),
    actual_raw_data_size_(bsdt.actual_raw_data_size_) {
  if (bsdt.pactual_raw_data_ != nullptr) {
    auto data_byte_size = actual_raw_data_size_ * sizeof(ElemT);
    pactual_raw_data_ = (ElemT *) malloc(data_byte_size);
    memcpy(pactual_raw_data_, bsdt.pactual_raw_data_, data_byte_size);
  }
}


/**
Assign a block sparse data tensor.

@param rhs Another block sparse data tensor.
*/
template <typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT> &
BlockSparseDataTensor<ElemT, QNT>::operator=(const BlockSparseDataTensor &rhs) {
  free(pactual_raw_data_);
  ten_rank = rhs.ten_rank;
  blk_shape = rhs.blk_shape;
  blk_multi_dim_offsets = rhs.blk_multi_dim_offsets;
  blk_idx_data_blk_map_ = rhs.blk_idx_data_blk_map_;
  pgqten_indexes = rhs.pgqten_indexes;
  actual_raw_data_size_ = rhs.actual_raw_data_size_;
  if (rhs.pactual_raw_data_ != nullptr) {
    auto data_byte_size = actual_raw_data_size_ * sizeof(ElemT);
    pactual_raw_data_ = (ElemT *) malloc(data_byte_size);
    memcpy(pactual_raw_data_, rhs.pactual_raw_data_, data_byte_size);
  } else {
    pactual_raw_data_ = nullptr;
  }
  return *this;
}


/// Destroy a block sparse data tensor.
template <typename ElemT, typename QNT>
BlockSparseDataTensor<ElemT, QNT>::~BlockSparseDataTensor(void) {
  RawDataFree_();
}


/**
Get an element using block coordinates and in-block data coordinates (in the
degeneracy space).

@param blk_coors_data_coors Block coordinates and in-block data coordinates pair.

@return The tensor element.
*/
template <typename ElemT, typename QNT>
ElemT BlockSparseDataTensor<ElemT, QNT>::ElemGet(
    const std::pair<CoorsT, CoorsT> &blk_coors_data_coors
) const {
  auto blk_idx_data_blk_it = blk_idx_data_blk_map_.find(
                                 BlkCoorsToBlkIdx(blk_coors_data_coors.first)
                             );
  if (blk_idx_data_blk_it == blk_idx_data_blk_map_.end()) {
    return 0.0;
  } else {
    size_t inblk_data_idx =
        blk_idx_data_blk_it->second.DataCoorsToInBlkDataIdx(
            blk_coors_data_coors.second
        );
    return *(
        pactual_raw_data_ + blk_idx_data_blk_it->second.data_offset + inblk_data_idx
    );
  }
}


/**
Set an element using block coordinates and in-block data coordinates (in the
degeneracy space).

@param blk_coors_data_coors Block coordinates and in-block data coordinates
       pair.

@param elem The value of the tensor element.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ElemSet(
    const std::pair<CoorsT, CoorsT> &blk_coors_data_coors,
    const ElemT elem
) {
  auto blk_idx_data_blk_it = blk_idx_data_blk_map_.find(
                                 BlkCoorsToBlkIdx(blk_coors_data_coors.first)
                             );
  if (blk_idx_data_blk_it == blk_idx_data_blk_map_.end()) {
    blk_idx_data_blk_it = DataBlkInsert(blk_coors_data_coors.first);
  }
  size_t inblk_data_idx = blk_idx_data_blk_it->second.DataCoorsToInBlkDataIdx(
                              blk_coors_data_coors.second
                          );
  *(pactual_raw_data_ + blk_idx_data_blk_it->second.data_offset + inblk_data_idx) = elem;
}


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
  blk_multi_dim_offsets = CalcMultiDimDataOffsets(blk_shape);

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


template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::StreamRead(std::istream &is) {
  size_t data_blk_num;
  is >> data_blk_num;
  for (size_t i = 0; i < data_blk_num; ++i) {
    CoorsT blk_coors(ten_rank);
    for (size_t j = 0; j < ten_rank; ++j) {
      is >> blk_coors[j];
    }
    DataBlkInsert(blk_coors, false);
  }
  Allocate();
  RawDataRead_(is);
}


template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::StreamWrite(std::ostream &os) const {
  os << blk_idx_data_blk_map_.size() << std::endl;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_) {
    for (auto &blk_coor : blk_idx_data_blk.second.blk_coors) {
      os << blk_coor << std::endl;
    }
  }
  RawDataWrite_(os);
}



/**
Re-calculate and reset the data offset of each data block in a BlkIdxDataBlkMap.

@param blk_idx_data_blk_map A block index <-> data block mapping.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ResetDataOffset(
    BlkIdxDataBlkMap &blk_idx_data_blk_map
) {
  size_t data_offset = 0;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map) {
    blk_idx_data_blk.second.data_offset = data_offset;
    data_offset += blk_idx_data_blk.second.size;
  }
}


/**
Equivalence check. Only check data, not quantum number information.

@param rhs The other BlockSparseDataTensor at the right hand side.

@return Equivalence check result.
*/
template <typename ElemT, typename QNT>
bool BlockSparseDataTensor<ElemT, QNT>::operator==(
    const BlockSparseDataTensor &rhs
) const {
  auto data_blk_size = blk_idx_data_blk_map_.size();
  if (data_blk_size != rhs.blk_idx_data_blk_map_.size()) { return false; }
  auto lhs_idx_blk_it = blk_idx_data_blk_map_.begin();
  auto rhs_idx_blk_it = rhs.blk_idx_data_blk_map_.begin();
  for (size_t i = 0; i < data_blk_size; ++i) {
    if (lhs_idx_blk_it->first != rhs_idx_blk_it->first) { return false; }
    lhs_idx_blk_it++;
    rhs_idx_blk_it++;
  }
  return ArrayEq(
      pactual_raw_data_, actual_raw_data_size_,
      rhs.pactual_raw_data_, rhs.actual_raw_data_size_
  );
}


/**
Release the raw data, set the pointer to null, set the size to 0.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataFree_(void) {
  free(pactual_raw_data_);
  pactual_raw_data_ = nullptr;
  actual_raw_data_size_ = 0;
}


/**
Directly set raw data point to nullptr and set actual raw data size to 0.

@note The memory may leak!!
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataDiscard_(void) {
  pactual_raw_data_ = nullptr;
  actual_raw_data_size_ = 0;
}


/**
Allocate memoery using a size.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataAlloc_(const size_t size) {
  free(pactual_raw_data_);
  pactual_raw_data_ = (ElemT *) malloc(size * sizeof(ElemT));
  actual_raw_data_size_ = size;
}


/**
Insert a subarray to the raw data array and decide whether initialize the memory
of the subarray.

@param offset Start data offset for inserting subarray.
@param size   The size of the subarray.
@param init   Whether initialize the inserted subarray to 0.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataInsert_(
    const size_t offset,
    const size_t size,
    const bool init
) {
  if (actual_raw_data_size_ == 0) {
    assert(offset == 0);
    assert(pactual_raw_data_ == nullptr);
    pactual_raw_data_ = (ElemT *) malloc(size * sizeof(ElemT));
    actual_raw_data_size_ = size;
  } else {
    size_t new_data_size = actual_raw_data_size_ + size;
    ElemT *new_pdata = (ElemT *) malloc(new_data_size * sizeof(ElemT));
    memcpy(new_pdata, pactual_raw_data_, offset * sizeof(ElemT));
    memcpy(
        new_pdata + (offset + size),
        pactual_raw_data_ + offset,
        (actual_raw_data_size_ - offset) * sizeof(ElemT)
    );
    free(pactual_raw_data_);
    pactual_raw_data_ = new_pdata;
    actual_raw_data_size_ = new_data_size;
  }

  if (init) {
    memset(pactual_raw_data_ + offset, 0, size * sizeof(ElemT));
  }
}


/**
Random set all the actual raw data to [0, 1].
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataRand_(void) {
  for (size_t i = 0; i < actual_raw_data_size_; ++i) {
    Rand(pactual_raw_data_[i]);
  }
}


/**
Tensor transpose for the 1D raw data array.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataTranspose_(
    const std::vector<RawDataTransposeTask> &raw_data_trans_tasks) {
  ElemT *ptransed_actual_raw_data = (ElemT *) malloc(actual_raw_data_size_ * sizeof(ElemT));
  for (auto &trans_task : raw_data_trans_tasks) {
    hp_numeric::TensorTranspose(
        trans_task.transed_order,
        trans_task.ten_rank,
        pactual_raw_data_ + trans_task.original_data_offset,
        trans_task.original_shape,
        ptransed_actual_raw_data + trans_task.transed_data_offset,
        trans_task.transed_shape
    );
  }
  free(pactual_raw_data_);
  pactual_raw_data_ = ptransed_actual_raw_data;
}


/**
Normalize the raw data array.

@return The norm before normalization.
*/
template <typename ElemT, typename QNT>
GQTEN_Double BlockSparseDataTensor<ElemT, QNT>::RawDataNormalize_(void) {
  GQTEN_Double norm2 = 0.0;
  for (size_t i = 0; i < actual_raw_data_size_; ++i) {
    norm2 += CalcScalarNorm2(pactual_raw_data_[i]);
  }
  auto norm = std::sqrt(norm2);
  for (size_t i = 0; i < actual_raw_data_size_; ++i) {
    pactual_raw_data_[i] /= norm;
  }
  return norm;
}


/**
Complex conjugate for raw data.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataConj_(void) {
  if (std::is_same<ElemT, GQTEN_Double>::value) {
    // Do nothing
  } else {
    for (size_t i = 0; i < actual_raw_data_size_; ++i) {
      pactual_raw_data_[i] = CalcConj(pactual_raw_data_[i]);
    }
  }
}


/**
Copy a piece of raw data from another place. You can decided whether add this
piece on the original one.

@param raw_data_copy_tasks Raw data copy task list.
@param psrc_raw_data The pointer to source data.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataCopy_(
    const std::vector<RawDataCopyTask> &raw_data_copy_tasks,
    const ElemT *psrc_raw_data
) {
  for (auto &task : raw_data_copy_tasks) {
    if (task.copy_and_add) {
      hp_numeric::VectorAddTo(
          psrc_raw_data + task.src_data_offset,
          task.src_data_size,
          pactual_raw_data_ + task.dest_data_offset
      );
    } else {
      memcpy(
          pactual_raw_data_ + task.dest_data_offset,
          psrc_raw_data + task.src_data_offset,
          task.src_data_size * sizeof(ElemT)
      );
    }
  }
}


/**
Duplicate a whole same size real raw data array from another place.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataDuplicateFromReal_(
    const GQTEN_Double *preal_raw_data_, const size_t size) {
  if (std::is_same<ElemT, GQTEN_Complex>::value) {
    hp_numeric::VectorRealToCplx(preal_raw_data_, size, pactual_raw_data_);
  } else {
    assert(false);    // TODO: To-be implemented!
  }
}


/**
Multiply the raw data by a scalar.

@param s A scalar.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataMultiplyByScalar_(
    const ElemT s
) {
  hp_numeric::VectorScale(pactual_raw_data_, actual_raw_data_size_, s);
}


/**
Read raw data from a stream.

@param is Input stream.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataRead_(std::istream &is) {
  is.seekg(1, std::ios::cur);    // Skip the line break.
  is.read((char *) pactual_raw_data_, actual_raw_data_size_ * sizeof(ElemT));
}


/**
Write raw data to a stream.

@param os Output stream.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataWrite_(std::ostream &os) const {
  os.write((char *) pactual_raw_data_, actual_raw_data_size_ * sizeof(ElemT));
  os << std::endl;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H */
