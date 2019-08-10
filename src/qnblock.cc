// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-07 20:26
* 
* Description: GraceQ/tensor project. Implementation details about quantum number block.
*/
#include "gqten/gqten.h"
#include "ten_trans.h"
#include "vec_hash.h"
#include "utils.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


QNBlock::QNBlock(const std::vector<QNSector> &init_qnscts) :
    QNSectorSet(init_qnscts) {
  ndim = qnscts.size(); 
  for (auto &qnsct : qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) { size *= shape[i]; }
    data_ = new double[size] ();    // Allocate memory and initialize to 0.
    data_offsets_ = CalcMultiDimDataOffsets(shape);
    qnscts_hash_ = QNSectorSet::Hash();
  }
}


QNBlock::QNBlock(const std::vector<const QNSector *> &pinit_qnscts) :
    QNSectorSet(pinit_qnscts) {
  ndim = qnscts.size(); 
  for (auto &qnsct : qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) {
      size *= shape[i];
    }
    data_ = new double[size];    // Allocate memory. NOT INITIALIZE TO ZERO!!!
    data_offsets_ = CalcMultiDimDataOffsets(shape);
    qnscts_hash_ = QNSectorSet::Hash();
  }
}


QNBlock::QNBlock(const QNBlock &qnblk) :
    QNSectorSet(qnblk),   // Use copy constructor of the base class.
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_) {
  data_ = new double[size];
  std::memcpy(data_, qnblk.data_, size * sizeof(double));
}


QNBlock &QNBlock::operator=(const QNBlock &rhs) {
  // Copy data.
  auto new_data = new double [rhs.size];
  std::memcpy(new_data, rhs.data_, rhs.size * sizeof(double));
  delete [] data_;
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


QNBlock::QNBlock(QNBlock &&qnblk) noexcept :
    QNSectorSet(qnblk),
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_),
    data_(qnblk.data_) {
  qnblk.data_ = nullptr;
}


QNBlock &QNBlock::operator=(QNBlock &&rhs) noexcept {
  // Move data.
  delete [] data_;
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


QNBlock::~QNBlock(void) {
  delete [] data_;
  data_ = nullptr;
}


// Block element getter.
const double &QNBlock::operator()(const std::vector<long> &coors) const {
  assert(coors.size() == ndim);
  auto offset = CalcEffOneDimArrayOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


// Block element setter.
double &QNBlock::operator()(const std::vector<long> &coors) {
  assert(coors.size() == ndim);
  auto offset = CalcEffOneDimArrayOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


size_t QNBlock::PartHash(const std::vector<long> &axes) const {
  auto selected_qnscts_ndim  = axes.size();
  std::vector<const QNSector *> pselected_qnscts(selected_qnscts_ndim);
  for (std::size_t i = 0; i < selected_qnscts_ndim; ++i) {
    pselected_qnscts[i] = &qnscts[axes[i]];
  }
  return VecPtrHasher(pselected_qnscts);
}


// Inplace operation.
void QNBlock::Random(void) {
  for (int i = 0; i < size; ++i) { data_[i] = double(rand()) / RAND_MAX; }
}


void QNBlock::Transpose(const std::vector<long> &transed_axes) {
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
  delete[] data_;
  data_ = new_data;
  shape = transed_shape;
  qnscts = transed_qnscts;
  data_offsets_ = transed_data_offsets_;
}


std::ifstream &bfread(std::ifstream &ifs, QNBlock &qnblk) {
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
    qnblk.data_ = new double[qnblk.size];
    ifs.read((char *) qnblk.data_, qnblk.size*sizeof(double));
  }
  return ifs;
}


std::ofstream &bfwrite(std::ofstream &ofs, const QNBlock &qnblk) {
  ofs << qnblk.ndim << std::endl;

  ofs << qnblk.size << std::endl;

  for (auto &order : qnblk.shape) { ofs << order << std::endl; }

  for (auto &qnsct : qnblk.qnscts) { bfwrite(ofs, qnsct); }

  for (auto &offset : qnblk.data_offsets_) { ofs << offset << std::endl; }

  ofs << qnblk.qnscts_hash_ << std::endl;

  if (qnblk.size != 0) {
    ofs.write((char *) qnblk.data_, qnblk.size*sizeof(double));
  }
  ofs << std::endl;
  return ofs;
}
} /* gqten */ 
