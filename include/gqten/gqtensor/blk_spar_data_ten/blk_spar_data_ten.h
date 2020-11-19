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


#include "gqten/framework/value_t.h"                      // CoorsT, ShapeT
#include "gqten/framework/bases/streamable.h"             // Streamable
#include "gqten/gqtensor/index.h"                         // IndexVec, CalcQNSctNumOfIdxs
#include "gqten/gqtensor/blk_spar_data_ten/data_blk.h"    // DataBlk
#include "gqten/utility/utils_inl.h"                      // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets, Rand

#include <map>        // map

#include <stdlib.h>     // malloc
#include <string.h>     // memcpy, memset


namespace gqten {


/**
Block sparse data tensor.

@tparam ElemT Type of the tensor element.
@tparam QNT   Type of the quantum number.
*/
template <typename ElemT, typename QNT>
//class BlockSparseDataTensor : public Streamable {
class BlockSparseDataTensor {
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

  // Data block level operations.
  typename BlkIdxDataBlkMap::iterator
  DataBlkCreate(const CoorsT &blk_coors, const bool alloc_mem = true);

  // Global level operations.
  void Clear(void);
  void Allocate(void);
  void Random(void);

  // Operators overload
  bool operator==(const BlockSparseDataTensor &) const;
  bool operator!=(const BlockSparseDataTensor &rhs) const {
    return !(*this == rhs);
  }

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

  // Raw data operations.
  void RawDataFree_(void);
  void RawDataAlloc_(const size_t);
  void RawDataInsert_(const size_t, const size_t, const bool init = false);
  void RawDataRand_(void);
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
    blk_shape(bsdt.blk_shape),
    blk_multi_dim_offsets(bsdt.blk_multi_dim_offsets),
    blk_idx_data_blk_map_(bsdt.blk_idx_data_blk_map_),
    pgqten_indexes(bsdt.pgqten_indexes),
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
    blk_idx_data_blk_it = DataBlkCreate(blk_coors_data_coors.first);
  }
  size_t inblk_data_idx = blk_idx_data_blk_it->second.DataCoorsToInBlkDataIdx(
                              blk_coors_data_coors.second
                          );
  *(pactual_raw_data_ + blk_idx_data_blk_it->second.data_offset + inblk_data_idx) = elem;
}


/**
Create a new data block. User can decide whether allocate the memory.

@param blk_coors Block coordinates of the new data block.
@param alloc_mem Whether allocate the memory and set them to 0.

@return An iterator for this new block index <-> data block pair.

@note You can only gap the memory allocation procedure when the raw data is empty.
*/
template <typename ElemT, typename QNT>
typename BlockSparseDataTensor<ElemT, QNT>::BlkIdxDataBlkMap::iterator
BlockSparseDataTensor<ElemT, QNT>::DataBlkCreate(
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
Clear all contents of this block sparse data tensor.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::Clear(void) {
  blk_idx_data_blk_map_.clear();
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
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H */
