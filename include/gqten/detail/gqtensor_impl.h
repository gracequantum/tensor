// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 18:29
* 
* Description: GraceQ/tensor project. Implementation details for U1 symmetric block sparse tensor class template.
*/
#include <assert.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <cmath>

#include "gqten/gqten.h"
#include "gqten/detail/utils_inl.h"

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


// Forward declarations.
QN CalcDiv(const QNSectorSet &, const std::vector<Index> &);

std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &);


template <typename ElemType>
GQTensor<ElemType>::GQTensor(const std::vector<Index> &idxes) : indexes(idxes) {
  for (auto &index : indexes) {
    auto size = 0;
    for (auto &qnsct : index.qnscts) { size += qnsct.dim; }
    shape.push_back(size);
  }
}


template <typename ElemType>
GQTensor<ElemType>::GQTensor(const GQTensor<ElemType> &gqtensor) :
    indexes(gqtensor.indexes),
    scalar(gqtensor.scalar),
    shape(gqtensor.shape) {
  for (auto &pblk : gqtensor.blocks_) {
    auto pnew_blk = new QNBlock<ElemType>(*pblk);
    blocks_.push_back(pnew_blk);
  }
}


template <typename ElemType>
GQTensor<ElemType> &GQTensor<ElemType>::operator=(const GQTensor &rhs) {
  for (auto pblk : blocks_) { delete pblk; }    // Delete old blocks.
  auto new_blk_num = rhs.blocks_.size();
  std::vector<QNBlock<ElemType> *> new_blks(new_blk_num);
  for (size_t i = 0; i < new_blk_num; ++i) {
    auto pnew_blk = new QNBlock<ElemType>(*rhs.blocks_[i]);
    new_blks[i] = pnew_blk;
  }
  blocks_ = new_blks;
  scalar  = rhs.scalar;
  indexes = rhs.indexes;
  shape = rhs.shape;
  return *this;
}


template <typename ElemType>
GQTensor<ElemType>::GQTensor(GQTensor &&gqtensor) noexcept :
    indexes(gqtensor.indexes),
    scalar(gqtensor.scalar),
    shape(gqtensor.shape),
    blocks_(std::move(gqtensor.blocks_)) {}


template <typename ElemType>
GQTensor<ElemType> &GQTensor<ElemType>::operator=(GQTensor &&rhs) noexcept {
  for (auto pblk : blocks_) { delete pblk; }
  blocks_ = std::move(rhs.blocks_);
  scalar = rhs.scalar;
  indexes = rhs.indexes;
  shape = rhs.shape;
  return *this;
}


template <typename ElemType>
GQTensor<ElemType>::~GQTensor(void) {
  for (auto &pblk : blocks_) { delete pblk; }
}


template <typename ElemType>
double GQTensor<ElemType>::Elem(const std::vector<long> &coors) const {
  auto blk_inter_offsets_and_blk_qnss = CalcTargetBlkInterOffsetsAndQNSS(coors);
  for (auto &pblk : blocks_) {
    if (pblk->qnscts == blk_inter_offsets_and_blk_qnss.blk_qnss) {
      return (*pblk)(blk_inter_offsets_and_blk_qnss.blk_inter_offsets);
    }
  }
  return 0.0;
}


template <typename ElemType>
double &GQTensor<ElemType>::operator()(const std::vector<long> &coors) {
  auto blk_inter_offsets_and_blk_qnss = CalcTargetBlkInterOffsetsAndQNSS(coors);
  for (auto &pblk : blocks_) {
    if (pblk->qnscts == blk_inter_offsets_and_blk_qnss.blk_qnss) {
      return (*pblk)(blk_inter_offsets_and_blk_qnss.blk_inter_offsets);
    }
  }
  QNBlock<ElemType> *pnew_block = new QNBlock<ElemType>(
                               blk_inter_offsets_and_blk_qnss.blk_qnss.qnscts);
  blocks_.push_back(pnew_block);
  return (*blocks_.back())(blk_inter_offsets_and_blk_qnss.blk_inter_offsets);
}


template <typename ElemType>
void GQTensor<ElemType>::Transpose(const std::vector<long> &transed_axes) {
  assert(transed_axes.size() == indexes.size());
  // Transpose indexes and shape.
  std::vector<Index> transed_indexes(indexes.size());
  std::vector<long> transed_shape(shape.size());
  for (size_t i = 0; i < transed_axes.size(); ++i) {
    transed_indexes[i] = indexes[transed_axes[i]];
    transed_shape[i] = shape[transed_axes[i]];
  }
  indexes = transed_indexes;
  shape = transed_shape;
  // Transpose blocks.
  for (auto &pblk : blocks_) {
    pblk->Transpose(transed_axes);
  }
}


template <typename ElemType>
void GQTensor<ElemType>::Random(const QN &div) {
  for (auto &pblk : blocks_) { delete pblk; }
  blocks_ = std::vector<QNBlock<ElemType> *>();
  for (auto &blk_qnss : BlkQNSSsIter()) {
    if (CalcDiv(blk_qnss.qnscts, indexes) == div) {
      QNBlock<ElemType> *block = new QNBlock<ElemType>(blk_qnss.qnscts);
      block->Random();
      blocks_.push_back(block);
    }
  }
}


template <typename ElemType>
double GQTensor<ElemType>::Normalize(void) {
  auto norm = Norm();
  for (auto &pblk : blocks_) {
    auto data = pblk->data();
    for (long i = 0; i < pblk->size; ++i) {
      data[i] = data[i] / norm;
    }
  }
  return norm;
}


// Operators Overload.
template <typename ElemType>
GQTensor<ElemType> GQTensor<ElemType>::operator+(const GQTensor &rhs) {
  auto added_t = GQTensor(indexes);
  for (auto &prhs_blk : rhs.cblocks()) {
    auto  has_blk = false;
    for (auto &lhs_blk : blocks_) {
      if (lhs_blk->QNSectorSetHash() == prhs_blk->QNSectorSetHash()) {
        assert(lhs_blk->size == prhs_blk->size);
        auto added_blk = new QNBlock<ElemType>(lhs_blk->qnscts);
        auto added_data = new double [lhs_blk->size];
        auto lhs_blk_data = lhs_blk->cdata();
        auto rhs_blk_data = prhs_blk->cdata();
        for (long i = 0; i < lhs_blk->size; ++i) {
          added_data[i] = lhs_blk_data[i] + rhs_blk_data[i];
        }
        added_blk->data() = added_data;
        added_t.blocks().push_back(added_blk);
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto added_blk = new QNBlock<ElemType>(*prhs_blk);
      added_t.blocks().push_back(added_blk);
    }
  }
  for (auto &plhs_blk : blocks_) {
    auto has_blk = false;
    for (auto &existed_blk : added_t.cblocks()) {
      if (existed_blk->QNSectorSetHash() == plhs_blk->QNSectorSetHash()) {
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto added_blk = new QNBlock<ElemType>(*plhs_blk);
      added_t.blocks().push_back(added_blk);
    }
  }
  return std::move(added_t);      // Although the "copy elision" will make sure
                                  // that no copy/move happened here, Use the
                                  // explicit move function to tell others be
                                  // careful about that.
}


template <typename ElemType>
GQTensor<ElemType> &GQTensor<ElemType>::operator+=(const GQTensor &rhs) {
  if (this->indexes.size() == 0) {
    assert(this->indexes == rhs.indexes);
    this->scalar += rhs.scalar;
    return *this;
  }

  for (auto &prhs_blk : rhs.cblocks()) {
    auto has_blk = false;
    for (auto &plhs_blk : blocks_) {
      if (plhs_blk->QNSectorSetHash() == prhs_blk->QNSectorSetHash()) {
        auto lhs_blk_data = plhs_blk->data();
        auto rhs_blk_data = prhs_blk->cdata();
        assert(plhs_blk->size == prhs_blk->size);
        for (long i = 0; i < prhs_blk->size; i++) {
          lhs_blk_data[i] += rhs_blk_data[i];
        }
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto pnew_blk = new QNBlock<ElemType>(*prhs_blk);
      blocks_.push_back(pnew_blk);
    }
  }
  return *this;
}


template <typename ElemType>
GQTensor<ElemType> GQTensor<ElemType>::operator-(void) const {
  auto minus_t = GQTensor(*this);
  for (auto &pblk : minus_t.blocks()) {
    auto data = pblk->data();
    for (long i = 0; i < pblk->size; ++i) {
      data[i] = -data[i];
    }
  }
  return std::move(minus_t);
}


template <typename ElemType>
bool GQTensor<ElemType>::operator==(const GQTensor &rhs) const {
  // Indexes check.
  if (indexes != rhs.indexes) {
    return false;
  }
  // Scalar check.
  if (indexes.size() == 0 && rhs.indexes.size() == 0 && scalar != rhs.scalar) {
    return false;
  }
  // Block number check.
  if (blocks_.size() != rhs.blocks_.size()) {
    return false;
  }
  // Blocks check.
  for (auto &plhs_blk : blocks_) {
    auto has_eq_blk = false;
    for (auto &prhs_blk : rhs.blocks_) {
      if (prhs_blk->QNSectorSetHash() == plhs_blk->QNSectorSetHash()) {
        if (!ArrayEq(
                 prhs_blk->cdata(), prhs_blk->size,
                 plhs_blk->cdata(), plhs_blk->size)) {
          return false;
        } else {
          has_eq_blk = true;
          break;
        }
      }
    }
    if (!has_eq_blk) {
      return false;
    }
  }
  return true;
}


// Iterators.
// Generate all coordinates. Cost so much. Be careful to use.
template <typename ElemType>
std::vector<std::vector<long>> GQTensor<ElemType>::CoorsIter(void) const {
  return GenAllCoors(shape);
}


// Private members.
template <typename ElemType>
BlkInterOffsetsAndQNSS GQTensor<ElemType>::CalcTargetBlkInterOffsetsAndQNSS(
    const std::vector<long> &coors) const {
  std::vector<long> blk_inter_offsets(coors.size());
  std::vector<QNSector> blk_qnss(coors.size());
  for (size_t i = 0; i < coors.size(); ++i) {
    auto inter_offset_and_qnsct = indexes[i].CoorInterOffsetAndQnsct(coors[i]);
    blk_inter_offsets[i] = coors[i] - inter_offset_and_qnsct.inter_offset;
    blk_qnss[i] = inter_offset_and_qnsct.qnsct;
  }
  return BlkInterOffsetsAndQNSS(blk_inter_offsets, blk_qnss);
}


template <typename ElemType>
std::vector<QNSectorSet> GQTensor<ElemType>::BlkQNSSsIter(void) const {
  std::vector<std::vector<QNSector>> v;
  for (auto &index : indexes) { v.push_back(index.qnscts); }
  auto s = CalcCartProd(v);
  std::vector<QNSectorSet> blk_qnss_set;
  for (auto &qnscts : s) { blk_qnss_set.push_back(QNSectorSet(qnscts)); }
  return blk_qnss_set;
}


template <typename ElemType>
double GQTensor<ElemType>::Norm(void) {
  double norm2 = 0.0; 
  for (auto &pblk : blocks_) {
    for (long i = 0; i < pblk->size; ++i) {
      norm2 += std::pow(pblk->data()[i], 2.0);
    }
  }
  return std::sqrt(norm2);
}


// Tensor operations.
template <typename ElemType>
GQTensor<ElemType> Dag(const GQTensor<ElemType> &t) {
  GQTensor<ElemType> dag_t(t);
  dag_t.Dag();
  return std::move(dag_t);
}


template <typename ElemType>
QN Div(const GQTensor<ElemType> &t) {
  auto blks = t.cblocks();
  auto blk_num = blks.size();
  if (blk_num == 0) {
    std::cout << "Tensor does not have a block. Retrun QN()." << std::endl;
    return QN();
  }
  QN div = CalcDiv(blks[0]->qnscts, t.indexes);
  for (size_t i = 1; i < blk_num; ++i) {
    auto blki_div = CalcDiv(blks[i]->qnscts, t.indexes);
    if (blki_div != div) {
      std::cout << "Tensor does not have a special divergence. Return QN()." << std::endl;
      return QN();
    }
  }
  return div;
}


template <typename ElemType>
GQTensor<ElemType> operator*(const GQTensor<ElemType> &t, const double &s) {
  auto muled_t = GQTensor<ElemType>(t);
  // For scalar case.
  if (muled_t.indexes.size() == 0) {
    muled_t.scalar *= s;
    return muled_t;
  }
  // For tensor case.
  for (auto &pblk : muled_t.blocks()) {
    auto data = pblk->data();
    for (long i = 0; i < pblk->size; ++i) {
      data[i]  = data[i] * s;
    }
  }
  return std::move(muled_t);
}


template <typename ElemType>
GQTensor<ElemType> operator*(const double &s, const GQTensor<ElemType> &t) {
  return std::move(t * s);
}


template <typename ElemType>
std::ifstream &bfread(std::ifstream &ifs, GQTensor<ElemType> &t) {
  long ndim;
  ifs >> ndim;
  t.indexes = std::vector<Index>(ndim);
  for (auto &idx : t.indexes) { bfread(ifs, idx); }
  t.shape = std::vector<long>(ndim);
  for (auto &order : t.shape) { ifs >> order; }
  ifs >> t.scalar;

  long blk_num;
  ifs >> blk_num;
  for (auto &pblk : t.blocks_) { delete pblk; }
  t.blocks_ = std::vector<QNBlock<ElemType> *>(blk_num);
  for (auto &pblk : t.blocks_) {
    pblk = new QNBlock<ElemType>();
    bfread(ifs, *pblk);
  }
  return ifs;
}


template <typename ElemType>
std::ofstream &bfwrite(std::ofstream &ofs, const GQTensor<ElemType> &t) {
  long ndim = t.indexes.size();
  ofs << ndim << std::endl;
  for (auto &idx : t.indexes) { bfwrite(ofs, idx); }
  for (auto &order : t.shape) { ofs << order << std::endl; }
  ofs << t.scalar << std::endl;

  long blk_num = t.blocks_.size();
  ofs << blk_num << std::endl;
  for (auto &pblk : t.blocks_) { bfwrite(ofs, *pblk); }
  return ofs;
}
} /* gqten */ 
