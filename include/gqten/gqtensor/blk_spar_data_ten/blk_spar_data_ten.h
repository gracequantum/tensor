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
#include "gqten/gqtensor/index.h"                                         // IndexVec, CalcQNSctNumOfIdxs
#include "gqten/gqtensor/blk_spar_data_ten/data_blk.h"                    // DataBlk
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operation_tasks.h"    // RawDataTransposeTask
#include "gqten/gqtensor/blk_spar_data_ten/data_blk_mat.h"                // IdxDataBlkMatMap
#include "gqten/utility/utils_inl.h"                                      // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets, Reorder, ArrayEq, VecMultiSelectElemts

#include <map>              // map
#include <unordered_map>    // unordered_map
#include <iostream>         // endl, istream, ostream

#include <stdlib.h>     // malloc, free
#include <string.h>     // memcpy, memset
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


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
  std::vector<RawDataCtrctTask> DataBlkGenForTenCtrct(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      const std::vector<std::vector<size_t>> &
  );
  void DataBlksInsert(
      const std::vector<size_t> &,
      const std::vector<CoorsT> &,
      const bool,
      const bool = false
  );
  void DataBlksInsert(
      const std::vector<CoorsT> &,
      const bool,
      const bool
  );

  void DataBlkCopyAndScale(
      const RawDataCopyAndScaleTask<ElemT> &,
      const ElemT *
  );

  std::map<size_t, DataBlkMatSvdRes<ElemT>> DataBlkDecompSVD(
      const IdxDataBlkMatMap<QNT> &
  ) const;

  void DataBlkCopySVDUdata(
      const CoorsT &, const size_t, const size_t,
      const size_t,
      const ElemT *, const size_t, const size_t,
      const std::vector<size_t> &
  );

  void DataBlkCopySVDVtData(
      const CoorsT &, const size_t, const size_t,
      const size_t,
      const ElemT *, const size_t, const size_t,
      const std::vector<size_t> &
  );

  // Global level operations
  void Clear(void);
  void Allocate(const bool init = false);

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
  void CtrctTwoBSDTAndAssignIn(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      std::vector<RawDataCtrctTask> &
  );
  void ConstructExpandedDataOnFirstIndex(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      const std::vector<bool> &,
      const std::map<size_t, size_t> &
  );

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
  bool IsScalar(void) const;

  /**
  Calculate block index from block coordinates.

  @param blk_coors Block coordinates.

  @return Corresponding block index.
  */
  size_t BlkCoorsToBlkIdx(const CoorsT &blk_coors) const {
    return CalcEffOneDimArrayOffset(blk_coors, blk_multi_dim_offsets_);
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
  static std::unordered_map<size_t, size_t> GenBlkIdxQNBlkInfoPartHashMap(
      const BlkIdxDataBlkMap &,
      const std::vector<size_t> &
  );


  /// Rank of the tensor.
  size_t ten_rank = 0;
  /// Block shape.
  ShapeT blk_shape;
  /// A pointer which point to the indexes of corresponding GQTensor.
  const IndexVec<QNT> *pgqten_indexes = nullptr;

private:
  /// Ordered map from block index to data block for existed blocks.
  BlkIdxDataBlkMap blk_idx_data_blk_map_;

  /// Block multi-dimension data offsets;
  std::vector<size_t> blk_multi_dim_offsets_;

  /**
  Size of the raw data in this block sparse data tensor. This size must equal to
  the sum of the size of each existed DataBlk. The only exception is the rank 0
  (scalar) case which number of DataBlk is always zero but raw_data_size_ can
  equals to 1.

  @note This variable will only be changed in DataBlk* member functions.
  */
  size_t raw_data_size_ = 0;

  /**
  Actual size of the raw data. This size must equal to the of pactual_raw_data_.

  @note This variable will only be changed in Constructors and RawData* member functions.
  */
  size_t actual_raw_data_size_ = 0;

  /**
  Pointer which point to the actual one-dimensional data array (the raw data).

  @note This variable will only be changed in Constructors and RawData* member functions.
  */
  ElemT *pactual_raw_data_ = nullptr;

  // Private data block operations.
  void DataBlkClear_(void);

  // Raw data operations.
  void RawDataFree_(void);
  void RawDataDiscard_(void);

  void RawDataAlloc_(const size_t, const bool init = false);
  void RawDataInsert_(const size_t, const size_t, const bool init = false);

  void RawDataCopy_(const std::vector<RawDataCopyTask> &, const ElemT *);
  void RawDataCopyAndScale_(
      const RawDataCopyAndScaleTask<ElemT> &,
      const ElemT *
  );
  void RawDataSetZeros_(const size_t, const size_t);
  void RawDataSetZeros_(const std::vector<RawDataSetZerosTask> &);
  void RawDataDuplicateFromReal_(const GQTEN_Double *, const size_t);

  void RawDataRand_(void);
  void RawDataTranspose_(const std::vector<RawDataTransposeTask> &);
  GQTEN_Double RawDataNormalize_(void);
  void RawDataConj_(void);

  void RawDataMultiplyByScalar_(const ElemT);
  void RawDataTwoMatMultiplyAndAssignIn_(
      const ElemT *,
      const ElemT *,
      const size_t,
      const size_t, const size_t, const size_t,
      const ElemT
  );

  // Related to tensor decomposition
  ElemT *RawDataGenDenseDataBlkMat_(
      const TenDecompDataBlkMat<QNT> &
  ) const ;

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
  blk_multi_dim_offsets_ = CalcMultiDimDataOffsets(blk_shape);
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
    blk_multi_dim_offsets_(bsdt.blk_multi_dim_offsets_),
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
  blk_multi_dim_offsets_ = rhs.blk_multi_dim_offsets_;
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
  assert(blk_coors_data_coors.first.size() == ten_rank);
  assert(
      blk_coors_data_coors.first.size() == blk_coors_data_coors.second.size()
  );

  // For scalar case
  if (IsScalar()) {
    if (actual_raw_data_size_ == 1) {
      return *pactual_raw_data_;
    } else {
      return 0.0;
    }
  }

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
  assert(blk_coors_data_coors.first.size() == ten_rank);
  assert(
      blk_coors_data_coors.first.size() == blk_coors_data_coors.second.size()
  );

  // For scalar case
  if (IsScalar()) {
    if (actual_raw_data_size_ == 0) {
      raw_data_size_ = 1;
      Allocate();
      *pactual_raw_data_ = elem;
    } else {
      *pactual_raw_data_ = elem;
    }
    return;
  }

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
Rank 0 (scalar) check.
*/
template <typename ElemT, typename QNT>
bool BlockSparseDataTensor<ElemT, QNT>::IsScalar(void) const {
#ifdef NDEBUG
  return (ten_rank == 0);
#else     // Do more extra check
  bool is_rank0 = (ten_rank == 0);
  if (is_rank0) {
    assert(blk_shape.empty());
    assert(blk_multi_dim_offsets_.empty());
    assert(blk_idx_data_blk_map_.empty());
    assert((raw_data_size_ == 0) || (raw_data_size_ == 1));
    assert((actual_raw_data_size_ == 0) || (actual_raw_data_size_ == 1));
  }
  return is_rank0;
#endif
}


/**
Read a BlockSparseDataTensor from a stream.
*/
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

  if (IsScalar()) { raw_data_size_ = 1; }
  Allocate();
  RawDataRead_(is);
}


/**
Write a BlockSparseDataTensor to a stream.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::StreamWrite(std::ostream &os) const {
  os << blk_idx_data_blk_map_.size() << std::endl;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_) {
    for (auto &blk_coor : blk_idx_data_blk.second.blk_coors) {
      os << blk_coor << std::endl;
    }
  }

  if (IsScalar() && actual_raw_data_size_ == 0) {     // Empty scalar case
    ElemT *pscalar0 = new ElemT();
    os.write((char *) pscalar0, sizeof(ElemT));
    os << std::endl;
    delete pscalar0;
  } else {
    RawDataWrite_(os);
  }
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
Generate block index map <-> quantum block info part hash value.

@param blk_idx_data_blk_map A block index <-> data block map.
@param axes Selected axes indexes for part hash.

@return Block index <-> part hash value map.
*/
template <typename ElemT, typename QNT>
std::unordered_map<size_t, size_t>
BlockSparseDataTensor<ElemT, QNT>::GenBlkIdxQNBlkInfoPartHashMap(
    const BlkIdxDataBlkMap &blk_idx_data_blk_map,
    const std::vector<size_t> &axes
) {
  std::unordered_map<size_t, size_t> blk_idx_qnblk_info_part_hash_map;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map) {
    blk_idx_qnblk_info_part_hash_map[
        blk_idx_data_blk.first
    ] = blk_idx_data_blk.second.GetQNBlkInfo().PartHash(axes);
  }
  return blk_idx_qnblk_info_part_hash_map;
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
  if (IsScalar() && rhs.IsScalar()) {
    return ElemGet({}) == rhs.ElemGet({});
  }

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
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_BLK_SPAR_DATA_TEN_H */


// Include other details
#include "gqten/gqtensor/blk_spar_data_ten/global_operations.h"
#include "gqten/gqtensor/blk_spar_data_ten/data_blk_operations.h"
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operations.h"
