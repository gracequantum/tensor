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


// Warning: the input must be an empty BSDT without any DataBlk
// TODO: Provide similar feature in the BSDT class
template <typename ElemT, typename QNT>
QnInfoHashBlkCoorsShapeMap GenQnInfoHashBlkCoorsShapeMap(
    const BlockSparseDataTensor<ElemT, QNT> &bsdt
) {
  BlockSparseDataTensor<ElemT, QNT> bsdt_copy(bsdt);
  for (auto &blk_coors : GenAllCoors(bsdt_copy.blk_shape)) {
    bsdt_copy.DataBlkInsert(blk_coors, false);
  }

  QnInfoHashBlkCoorsShapeMap qninfohash_blkcoors_shape_map;
  for (auto &blk_idx_data_blk : bsdt_copy.GetBlkIdxDataBlkMap()) {
    auto data_blk = blk_idx_data_blk.second;
    qninfohash_blkcoors_shape_map[
        data_blk.GetQNBlkInfo().QnHash()
    ] = std::make_pair(data_blk.blk_coors, data_blk.shape);
  }

  return qninfohash_blkcoors_shape_map;
}


/**
Construct tensor expansion data from corresponding BSDTs.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ConstructExpandedDataFrom(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b
) {
  auto qninfohash_blkcoors_shape_map = GenQnInfoHashBlkCoorsShapeMap(*this);
  auto blk_idx_data_blk_map_a = bsdt_a.GetBlkIdxDataBlkMap();
  std::vector<size_t> tensor_embed_offsets_a(ten_rank, 0);
  std::vector<size_t> bsdt_a_existed_blk_qnhashs;
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_a) {
    auto data_blk_a = blk_idx_data_blk.second;
    auto qnhash = data_blk_a.GetQNBlkInfo().QnHash();
    bsdt_a_existed_blk_qnhashs.push_back(qnhash);
    assert(
        qninfohash_blkcoors_shape_map.find(qnhash) !=
        qninfohash_blkcoors_shape_map.end()
    );
    auto blk_coors_shape_pair = qninfohash_blkcoors_shape_map[qnhash];
    auto blk_idx_data_blk_it = DataBlkInsert(blk_coors_shape_pair.first);
    auto data_blk_expanded = blk_idx_data_blk_it->second;
    RawDataEmbed_(
        bsdt_a.GetActualRawDataPtr(),
        data_blk_a,
        data_blk_expanded,
        tensor_embed_offsets_a
    );
  }
  auto blk_idx_data_blk_map_b = bsdt_b.GetBlkIdxDataBlkMap();
  for (auto &blk_idx_data_blk : blk_idx_data_blk_map_b) {
    auto data_blk_b = blk_idx_data_blk.second;
    auto qnhash = data_blk_b.GetQNBlkInfo().QnHash();
    assert(
        qninfohash_blkcoors_shape_map.find(qnhash) !=
        qninfohash_blkcoors_shape_map.end()
    );
    auto blk_coors_shape_pair = qninfohash_blkcoors_shape_map[qnhash];
    // Calculate embed offsets
    std::vector<size_t> tensor_embed_offsets_b;
    tensor_embed_offsets_b.reserve(ten_rank);
    for (size_t i = 0; i < ten_rank; ++i) {
      tensor_embed_offsets_b.push_back(
          blk_coors_shape_pair.second[i] - data_blk_b.shape[i]
      );
    }
    auto bsdt_a_blk_qnhash_it = std::find(
                                    bsdt_a_existed_blk_qnhashs.begin(),
                                    bsdt_a_existed_blk_qnhashs.end(),
                                    qnhash
                                );
    // If the data block is existed
    if (bsdt_a_blk_qnhash_it != bsdt_a_existed_blk_qnhashs.end()) {
      auto data_blk_expanded = GetBlkIdxDataBlkMap().at(
                                   BlkCoorsToBlkIdx(blk_coors_shape_pair.first)
                               );
      RawDataEmbed_(
          bsdt_b.GetActualRawDataPtr(),
          data_blk_b,
          data_blk_expanded,
          tensor_embed_offsets_b
      );
    } else {
      auto blk_idx_data_blk_it = DataBlkInsert(blk_coors_shape_pair.first);
      auto data_blk_expanded = blk_idx_data_blk_it->second;
      RawDataEmbed_(
          bsdt_b.GetActualRawDataPtr(),
          data_blk_b,
          data_blk_expanded,
          tensor_embed_offsets_b
      );
    }
  }
}


/**
Construct tensor expansion data over the last index,  from corresponding BSDTs.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::ConstructExpandedDataOnFirstIndex(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    std::vector<bool>  is_a_idx_qnsct_expanded,
    std::map<size_t, size_t> expanded_idx_qnsct_coor_mapto_b_idx_qnsct_coor
) {

  const auto &blk_idx_data_blk_map_a=bsdt_a.GetBlkIdxDataBlkMap();
  auto blk_idx_data_blk_map_b=bsdt_b.GetBlkIdxDataBlkMap(); //We hope here is not a reference&

  std::map<size_t, int> blk_idx_expand_mapto_blk_map_a, blk_idx_expand_mapto_blk_map_b;
  // The new generated blk_data's index map to original blk_data index,
  // if no corresponding original blk_data but need filled with zero, label {blk_data_idx, -1};
  // if neither corresponding original blk_data nor filled with zero, no the pair.


  // First we construct the new blk_idx_data_blk_map
  for(auto &blk_idx_data_blk_pair_a : blk_idx_data_blk_map_a){
    size_t blk_coor_in_first_idx = blk_idx_data_blk_pair_a.second.blk_coors[0];
    size_t blk_idx_a = blk_idx_data_blk_pair_a.first;
    size_t blk_idx = blk_idx_a;
    if(!is_a_idx_qnsct_expanded[blk_coor_in_first_idx]) {
      blk_idx_data_blk_map_.insert(blk_idx_data_blk_pair_a);
      blk_idx_expand_mapto_blk_map_a[ blk_idx ] = blk_idx_a;
    }else{
      blk_idx_data_blk_map_[blk_idx_data_blk_pair_a.first]=DataBlk<QNT>(blk_idx_data_blk_pair_a.second.blk_coors, *pgqten_indexes);
      blk_idx_expand_mapto_blk_map_a[ blk_idx ] = blk_idx_a;

      std::vector<size_t> blk_coors_b = blk_idx_data_blk_pair_a.second.blk_coors;
      blk_coors_b[0]=expanded_idx_qnsct_coor_mapto_b_idx_qnsct_coor[blk_coor_in_first_idx];
      size_t blk_idx_b=bsdt_b.BlkCoorsToBlkIdx(blk_coors_b);
      auto pdata_blk_b=blk_idx_data_blk_map_b.find(blk_idx_b);
      if(pdata_blk_b!=blk_idx_data_blk_map_b.end()){
        blk_idx_expand_mapto_blk_map_b[blk_idx ] = pdata_blk_b->first;
        blk_idx_data_blk_map_b.erase( pdata_blk_b );
      }else{
        blk_idx_expand_mapto_blk_map_b[blk_idx ] = -1;
      }
    }
  }

  std::map<size_t, size_t> b_idx_qnsct_coor_mapto_expanded_idx_qnsct_coor;
  for(auto &elem: expanded_idx_qnsct_coor_mapto_b_idx_qnsct_coor ){
    b_idx_qnsct_coor_mapto_expanded_idx_qnsct_coor[elem.second]=elem.first;
  }
  for(auto &blk_idx_data_blk_pair_b : blk_idx_data_blk_map_b){
    size_t blk_coor_in_first_idx_b = blk_idx_data_blk_pair_b.second.blk_coors[0];
    size_t blk_coor_in_first_idx_expand = b_idx_qnsct_coor_mapto_expanded_idx_qnsct_coor[blk_coor_in_first_idx_b];
    std::vector <size_t> blk_coors = blk_idx_data_blk_pair_b.second.blk_coors;
    blk_coors[0] = blk_coor_in_first_idx_expand;
    //Generate the DataBlk
    size_t blk_idx = BlkCoorsToBlkIdx(blk_coors);
    blk_idx_data_blk_map_.insert(std::make_pair(blk_idx, DataBlk<QNT>(blk_coors, *pgqten_indexes)));
    blk_idx_expand_mapto_blk_map_b[blk_idx] = blk_idx_data_blk_pair_b.first;
    if (blk_coor_in_first_idx_expand < is_a_idx_qnsct_expanded.size()) {
      blk_idx_expand_mapto_blk_map_a[blk_idx] = -1;
    }
  }


  //allocate memory
  raw_data_size_ = 0;
  for(auto &blk_idx_data_blk_pair : blk_idx_data_blk_map_ ){
    blk_idx_data_blk_pair.second.data_offset= raw_data_size_;
    raw_data_size_ += blk_idx_data_blk_pair.second.size;
  }
  RawDataAlloc_(raw_data_size_, false);

  //copy and write raw data
  blk_idx_data_blk_map_b=bsdt_b.GetBlkIdxDataBlkMap(); // regenerate it
  std::vector<RawDataCopyTask> raw_data_copy_tasks_fromA,raw_data_copy_tasks_fromB;

  size_t raw_data_write_off_set=0;
  for(auto &blk_idx_data_blk_pair : blk_idx_data_blk_map_ ){
    size_t blk_idx = blk_idx_data_blk_pair.first;
    DataBlk<QNT> data_blk = blk_idx_data_blk_pair.second;
    if(blk_idx_expand_mapto_blk_map_a.find( blk_idx ) != blk_idx_expand_mapto_blk_map_a.end()){
      int blk_idx_a = blk_idx_expand_mapto_blk_map_a[blk_idx];
      if(blk_idx_a!=-1) {
        assert(blk_idx_a==blk_idx);
        auto pblk_idx_data_blk_pair_a = blk_idx_data_blk_map_a.find( static_cast<size_t>(blk_idx_a) );
        size_t raw_data_offset_in_a = pblk_idx_data_blk_pair_a->second.data_offset;

        RawDataCopyTask task=RawDataCopyTask(pblk_idx_data_blk_pair_a->second.blk_coors,
                        pblk_idx_data_blk_pair_a->second.data_offset, //raw_data_offset_in_a
                        pblk_idx_data_blk_pair_a->second.size);
        task.dest_data_offset = raw_data_write_off_set;
        assert(raw_data_write_off_set == data_blk.data_offset);
        raw_data_copy_tasks_fromA.push_back(task);
//
//        memcpy(pactual_raw_data_ + raw_data_write_off_set,
//               bsdt_a.GetActualRawDataPtr() + raw_data_offset_in_a,
//               (pblk_idx_data_blk_pair_a->second.size)*sizeof(ElemT));
        raw_data_write_off_set +=  pblk_idx_data_blk_pair_a->second.size;
      }else{
        size_t filled_zero_elem_number = data_blk.size-blk_idx_data_blk_map_b[blk_idx_expand_mapto_blk_map_b[blk_idx]].size;
        assert(raw_data_write_off_set == data_blk.data_offset);
        RawDataSetZeros_(raw_data_write_off_set, filled_zero_elem_number);
        raw_data_write_off_set += filled_zero_elem_number;
      }
    }


    if(blk_idx_expand_mapto_blk_map_b.find(blk_idx) !=  blk_idx_expand_mapto_blk_map_b.end()){
      int blk_idx_b = blk_idx_expand_mapto_blk_map_b[blk_idx];
      if(blk_idx_b!=-1) {
        size_t raw_data_offset_in_b = blk_idx_data_blk_map_b[blk_idx_b].data_offset;

        RawDataCopyTask task=RawDataCopyTask(blk_idx_data_blk_map_b[blk_idx_b].blk_coors,
                                             blk_idx_data_blk_map_b[blk_idx_b].data_offset, //raw_data_offset_in_b
                                             blk_idx_data_blk_map_b[blk_idx_b].size);
        task.dest_data_offset = raw_data_write_off_set;
        raw_data_copy_tasks_fromB.push_back(task);
        assert(raw_data_write_off_set == data_blk.data_offset + data_blk.size - blk_idx_data_blk_map_b[blk_idx_b].size );
//        memcpy(pactual_raw_data_ + raw_data_write_off_set,
//               bsdt_b.GetActualRawDataPtr() + raw_data_offset_in_b,
//               blk_idx_data_blk_map_b[blk_idx_b].size*sizeof(ElemT));
        raw_data_write_off_set += blk_idx_data_blk_map_b[blk_idx_b].size;
      }else{
        int blk_idx_a=blk_idx_expand_mapto_blk_map_a[blk_idx];
        assert(blk_idx_a == blk_idx);
        auto pblk_idx_data_blk_pair_a = blk_idx_data_blk_map_a.find( static_cast<size_t>(blk_idx_a) );
        size_t filled_zero_elem_number = data_blk.size-pblk_idx_data_blk_pair_a->second.size;
        assert(raw_data_write_off_set == data_blk.data_offset + pblk_idx_data_blk_pair_a->second.size );
        RawDataSetZeros_(raw_data_write_off_set, filled_zero_elem_number);
        raw_data_write_off_set += filled_zero_elem_number;
      }
    }
  }

  assert(raw_data_write_off_set == actual_raw_data_size_);
  RawDataCopy_(raw_data_copy_tasks_fromA , bsdt_a.GetActualRawDataPtr() );
  RawDataCopy_(raw_data_copy_tasks_fromB , bsdt_b.GetActualRawDataPtr() );
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
