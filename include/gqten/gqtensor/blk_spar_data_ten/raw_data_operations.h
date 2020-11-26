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
#include "gqten/utility/utils_inl.h"                                      // Rand, CalcScalarNorm2, CalcConj

#include <iostream>     // endl, istream, ostream
#include <cmath>        // sqrt

#include <stdlib.h>     // malloc, free
#include <string.h>     // memcpy, memset
#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


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
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATIONS_H */
