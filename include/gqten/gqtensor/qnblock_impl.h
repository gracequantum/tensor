// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 15:24
* 
* Description: GraceQ/tensor project. Implementation details for quantum number class template.
*/
#include <assert.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>

#include "gqten/gqten.h"
#include "gqten/vec_hash.h"
#include "gqten/utils_inl.h"

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


// Forward declarations.
std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &);
GQTEN_Double *DenseTensorTranspose(
    const GQTEN_Double *,
    const long,
    const long,
    const std::vector<long> &,
    const std::vector<long> &);


GQTEN_Complex *DenseTensorTranspose(
    const GQTEN_Complex *,
    const long,
    const long,
    const std::vector<long> &,
    const std::vector<long> &);


template <typename ElemType>
QNBlock<ElemType>::QNBlock(const std::vector<QNSector> &init_qnscts) :
    QNSectorSet(init_qnscts) {
  ndim = qnscts.size(); 
  for (auto &qnsct : qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) { size *= shape[i]; }
    data_ = (ElemType *)calloc(size, sizeof(ElemType));    // Allocate memory and initialize to 0.
    data_offsets_ = CalcMultiDimDataOffsets(shape);
    qnscts_hash_ = QNSectorSet::Hash();
  }
}


template <typename ElemType>
QNBlock<ElemType>::QNBlock(const std::vector<const QNSector *> &pinit_qnscts) :
    QNSectorSet(pinit_qnscts) {

#ifdef GQTEN_TIMING_MODE
    Timer qnblk_intra_construct_timer("qnblk_intra_construct");
    qnblk_intra_construct_timer.Restart();
#endif

  ndim = qnscts.size(); 
  for (auto &qnsct : qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) {
      size *= shape[i];
    }

#ifdef GQTEN_TIMING_MODE
    Timer qnblk_intra_construct_new_data_timer("qnblk_intra_construct_new_data");
    qnblk_intra_construct_new_data_timer.Restart();
#endif

    data_ = (ElemType *)malloc(size * sizeof(ElemType));      // Allocate memory. NOT INITIALIZE TO ZERO!!!

#ifdef GQTEN_TIMING_MODE
    qnblk_intra_construct_new_data_timer.PrintElapsed(8);
#endif

    data_offsets_ = CalcMultiDimDataOffsets(shape);
    qnscts_hash_ = QNSectorSet::Hash();
  }

#ifdef GQTEN_TIMING_MODE
    qnblk_intra_construct_timer.PrintElapsed(8);
#endif

}


template <typename ElemType>
QNBlock<ElemType>::QNBlock(const QNBlock &qnblk) :
    QNSectorSet(qnblk),   // Use copy constructor of the base class.
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_) {
  data_ = (ElemType *)malloc(size * sizeof(ElemType));
  std::memcpy(data_, qnblk.data_, size * sizeof(ElemType));
}


template <typename ElemType>
QNBlock<ElemType> &QNBlock<ElemType>::operator=(const QNBlock &rhs) {
  // Copy data.
  auto new_data = (ElemType *)malloc(rhs.size * sizeof(ElemType));
  std::memcpy(new_data, rhs.data_, rhs.size * sizeof(ElemType));
  free(data_);
  data_ = new_data;
  // Copy other members.
  qnscts = rhs.qnscts;    // For the base class.
  ndim = rhs.ndim;
  shape = rhs.shape;
  size = rhs.size;
  data_offsets_ = rhs.data_offsets_;
  qnscts_hash_ = rhs.qnscts_hash_;
  return *this;
}


template <typename ElemType>
QNBlock<ElemType>::QNBlock(QNBlock &&qnblk) noexcept :
    QNSectorSet(qnblk),
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_),
    data_(qnblk.data_) {
  qnblk.data_ = nullptr;
}


template <typename ElemType>
QNBlock<ElemType> &QNBlock<ElemType>::operator=(QNBlock &&rhs) noexcept {
  // Move data.
  free(data_);
  data_ = rhs.data_;
  rhs.data_ = nullptr;
  // Copy other members.
  qnscts = rhs.qnscts;    // For the base class.
  ndim = rhs.ndim;
  shape = rhs.shape;
  size = rhs.size;
  data_offsets_ = rhs.data_offsets_;
  qnscts_hash_ = rhs.qnscts_hash_;
  return *this;
}


template <typename ElemType>
QNBlock<ElemType>::~QNBlock(void) {
  free(data_);
  data_ = nullptr;
}


// Block element getter.
template <typename ElemType>
const ElemType &QNBlock<ElemType>::operator()(
    const std::vector<long> &coors) const {
  assert(coors.size() == ndim);
  auto offset = CalcEffOneDimArrayOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


// Block element setter.
template <typename ElemType>
ElemType &QNBlock<ElemType>::operator()(const std::vector<long> &coors) {
  assert(coors.size() == ndim);
  auto offset = CalcEffOneDimArrayOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


template <typename ElemType>
size_t QNBlock<ElemType>::PartHash(const std::vector<long> &axes) const {
  auto selected_qnscts_ndim  = axes.size();
  std::vector<const QNSector *> pselected_qnscts(selected_qnscts_ndim);
  for (std::size_t i = 0; i < selected_qnscts_ndim; ++i) {
    pselected_qnscts[i] = &qnscts[axes[i]];
  }
  return VecPtrHasher(pselected_qnscts);
}


// Inplace operation.
template <typename ElemType>
void QNBlock<ElemType>::Random(void) {
  for (int i = 0; i < size; ++i) { Rand(data_[i]); }
}


template <typename ElemType>
void QNBlock<ElemType>::Transpose(const std::vector<long> &transed_axes) {
  std::vector<QNSector> transed_qnscts(ndim);
  std::vector<long> transed_shape(ndim);
  for (long i = 0; i < ndim; ++i) {
    transed_qnscts[i] = qnscts[transed_axes[i]];
    transed_shape[i] = transed_qnscts[i].dim;
  }
  auto transed_data_offsets_ = CalcMultiDimDataOffsets(transed_shape);
  auto new_data = DenseTensorTranspose(
                      data_,
                      ndim, size, shape,
                      transed_axes);
  free(data_);
  data_ = new_data;
  shape = transed_shape;
  qnscts = transed_qnscts;
  data_offsets_ = transed_data_offsets_;
}


template <typename ElemType>
std::ifstream &bfread(std::ifstream &ifs, QNBlock<ElemType> &qnblk) {
  ifs >> qnblk.ndim;

  ifs >> qnblk.size;

  qnblk.shape = std::vector<long>(qnblk.ndim);
  for (auto &order : qnblk.shape) { ifs >> order; }

  qnblk.qnscts = std::vector<QNSector>(qnblk.ndim);
  for (auto &qnsct : qnblk.qnscts) { bfread(ifs, qnsct); }

  qnblk.data_offsets_ = std::vector<long>(qnblk.ndim);
  for (auto &offset : qnblk.data_offsets_) { ifs >> offset; }

  ifs >> qnblk.qnscts_hash_;

  ifs.seekg(1, std::ios::cur);    // Skip the line break.

  if (qnblk.size != 0) {
    qnblk.data_ = (ElemType *)malloc(qnblk.size * sizeof(ElemType));
    ifs.read((char *) qnblk.data_, qnblk.size*sizeof(ElemType));
  }
  return ifs;
}


template <typename ElemType>
std::ofstream &bfwrite(std::ofstream &ofs, const QNBlock<ElemType> &qnblk) {
  ofs << qnblk.ndim << std::endl;

  ofs << qnblk.size << std::endl;

  for (auto &order : qnblk.shape) { ofs << order << std::endl; }

  for (auto &qnsct : qnblk.qnscts) { bfwrite(ofs, qnsct); }

  for (auto &offset : qnblk.data_offsets_) { ofs << offset << std::endl; }

  ofs << qnblk.qnscts_hash_ << std::endl;

  if (qnblk.size != 0) {
    ofs.write((char *) qnblk.data_, qnblk.size*sizeof(ElemType));
  }
  ofs << std::endl;
  return ofs;
}
} /* gqten */ 
