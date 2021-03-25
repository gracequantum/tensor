// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-26 21:11
*
* Description: GraceQ/tensor project. Raw data operation member functions in
* BlockSparseDataTensor.
*/

/**
@file raw_data_operations.h
@brief Raw data operation member functions in BlockSparseDataTensor.
*/
#ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H
#define GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H


#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "gqten/framework/value_t.h"                                      // CoorsT, ShapeT
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operation_tasks.h"    // RawDataTransposeTask
#include "gqten/framework/hp_numeric/ten_trans.h"                         // TensorTranspose
#include "gqten/framework/hp_numeric/blas_level1.h"                       // VectorAddTo
#include "gqten/framework/hp_numeric/blas_level3.h"                       // MatMultiply
#include "gqten/framework/hp_numeric/lapack.h"                            // MatSVD
#include "gqten/utility/utils_inl.h"                                      // Rand, CalcScalarNorm2, CalcConj, SubMatMemCpy

#include <iostream>     // endl, istream, ostream
#include <cmath>        // sqrt

#include <stdlib.h>     // malloc, free, calloc
#include <string.h>     // memcpy, memset
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


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

@param init Whether initialize the memory to 0.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataAlloc_(
    const size_t size,
    const bool init
) {
  free(pactual_raw_data_);
  if (!init) {
    pactual_raw_data_ = (ElemT *) malloc(size * sizeof(ElemT));
  } else {
    pactual_raw_data_ = (ElemT *) calloc(size, sizeof(ElemT));
  }
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
Copy and scale a piece of raw data from another place. You can decided whether
add this piece on the original one.

@param raw_data_copy_tasks Raw data copy task list.
@param psrc_raw_data The pointer to source data.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataCopyAndScale_(
    const RawDataCopyAndScaleTask<ElemT> &raw_data_copy_and_scale_task,
    const ElemT *psrc_raw_data
) {
  auto dest_data_offset = blk_idx_data_blk_map_[
                              BlkCoorsToBlkIdx(
                                  raw_data_copy_and_scale_task.dest_blk_coors
                              )
                          ].data_offset;
  if (raw_data_copy_and_scale_task.copy_and_add) {
    hp_numeric::VectorAddTo(
        psrc_raw_data + raw_data_copy_and_scale_task.src_data_offset,
        raw_data_copy_and_scale_task.src_data_size,
        pactual_raw_data_ + dest_data_offset,
        raw_data_copy_and_scale_task.coef
    );
  } else {
    hp_numeric::VectorScaleCopy(
        psrc_raw_data + raw_data_copy_and_scale_task.src_data_offset,
        raw_data_copy_and_scale_task.src_data_size,
        pactual_raw_data_ + dest_data_offset,
        raw_data_copy_and_scale_task.coef
    );
  }
}


/**
Embed a data block from otherwhere to a data block in this BSDT. A series of
offsets at each dimension can be set.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataEmbed_(
    const ElemT *psrc_raw_data,
    const DataBlk<QNT> &src_data_blk,
    const DataBlk<QNT> &dest_data_blk,
    const CoorsT &tensor_embed_offsets
) {
  if (src_data_blk.shape == dest_data_blk.shape) {
    assert(tensor_embed_offsets == std::vector<size_t>(ten_rank, 0));
    RawDataCopyTask ten_emb_as_copy_datacopytask(
        src_data_blk.blk_coors,
        src_data_blk.data_offset,
        src_data_blk.size
    );
    ten_emb_as_copy_datacopytask.dest_data_offset = dest_data_blk.data_offset;
    RawDataCopy_({ten_emb_as_copy_datacopytask}, psrc_raw_data);
  } else {
    for (auto src_data_coors : GenAllCoors(src_data_blk.shape)) {
      auto src_inblk_data_idx = src_data_blk.DataCoorsToInBlkDataIdx(
                                    src_data_coors
                                );
      auto dest_data_coors = CoorsAdd(src_data_coors, tensor_embed_offsets);
      auto dest_inblk_data_idx = dest_data_blk.DataCoorsToInBlkDataIdx(
                                     dest_data_coors
                                 );
      pactual_raw_data_[dest_data_blk.data_offset + dest_inblk_data_idx] =
      psrc_raw_data[src_data_blk.data_offset + src_inblk_data_idx];
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
  if (actual_raw_data_size_ != 0) {
    hp_numeric::VectorScale(pactual_raw_data_, actual_raw_data_size_, s);
  }
}


/**
Multiply two matrices and assign in.
*/
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::RawDataTwoMatMultiplyAndAssignIn_(
    const ElemT *a,
    const ElemT *b,
    const size_t c_data_offset,
    const size_t m, const size_t k, const size_t n,
    const ElemT beta
) {
  assert(actual_raw_data_size_ != 0);
  hp_numeric::MatMultiply(
      a,
      b,
      m, k, n,
      beta,
      pactual_raw_data_ + c_data_offset
  );
}


template <typename ElemT, typename QNT>
ElemT *BlockSparseDataTensor<ElemT, QNT>::RawDataGenDenseDataBlkMat_(
    const TenDecompDataBlkMat<QNT> &data_blk_mat
) const {
  auto rows = data_blk_mat.rows;
  auto cols = data_blk_mat.cols;
  ElemT *mat = (ElemT *) calloc(rows * cols, sizeof(ElemT));
  for (auto &elem : data_blk_mat.elems) {
    auto i = elem.first[0];
    auto j = elem.first[1];
    auto row_offset = std::get<1>(data_blk_mat.row_scts[i]);
    auto col_offset = std::get<1>(data_blk_mat.col_scts[j]);
    auto m = std::get<2>(data_blk_mat.row_scts[i]);
    auto n = std::get<2>(data_blk_mat.col_scts[j]);
    auto blk_idx_in_bsdt = elem.second;
    auto sub_mem_begin = pactual_raw_data_ +
                         blk_idx_data_blk_map_.at(blk_idx_in_bsdt).data_offset;
    SubMatMemCpy(
        rows, cols,
        row_offset, col_offset,
        m, n, sub_mem_begin,
        mat
    );
  }
  return mat;
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
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H */
