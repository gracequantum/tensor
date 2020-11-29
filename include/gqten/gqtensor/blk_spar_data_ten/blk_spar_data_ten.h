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
#include "gqten/utility/utils_inl.h"                                      // CalcEffOneDimArrayOffset, CalcMultiDimDataOffsets, Reorder, ArrayEq, VecMultiSelectElemts

#include <map>              // map
#include <unordered_map>    // unordered_map
#include <iostream>         // endl, istream, ostream

#include <stdlib.h>     // malloc, free
#include <string.h>     // memcpy, memset
#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


// Some helpers
// For tensor contraction
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
  ElemT CtrctTwoBSDTAndAssignIn(
      const BlockSparseDataTensor &,
      const BlockSparseDataTensor &,
      std::vector<RawDataCtrctTask> &
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
  the sum of the size of each existed DataBlk.

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

  void RawDataAlloc_(const size_t);
  void RawDataInsert_(const size_t, const size_t, const bool init = false);

  void RawDataCopy_(const std::vector<RawDataCopyTask> &, const ElemT *);
  void RawDataDuplicateFromReal_(const GQTEN_Double *, const size_t);

  void RawDataRand_(void);
  void RawDataTranspose_(const std::vector<RawDataTransposeTask> &);
  GQTEN_Double RawDataNormalize_(void);
  void RawDataConj_(void);

  void RawDataMultiplyByScalar_(const ElemT);
  ElemT RawDataTwoMatMultiplyAndAssignIn_(
      const ElemT *,
      const ElemT *,
      const size_t,
      const size_t, const size_t, const size_t,
      const ElemT
  );

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
  bool c_is_scalar;
  if (saved_axes_set[0].empty() && saved_axes_set[1].empty()) {
    c_is_scalar = true;
  } else {
    c_is_scalar = false;
  }
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
    }
  }

  return raw_data_ctrct_tasks;
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


// Include other implementation details
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operations.h"
#include "gqten/gqtensor/blk_spar_data_ten/global_operations.h"
