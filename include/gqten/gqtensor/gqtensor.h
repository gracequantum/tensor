// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-13 16:18
*
* Description: GraceQ/tensor project. Symmetry-blocked sparse tensor.
*/

/**
@file gqtensor.h
@brief Symmetry-blocked sparse tensor.
*/
#ifndef GQTEN_GQTENSOR_GQTENSOR_H
#define GQTEN_GQTENSOR_GQTENSOR_H


#include "gqten/framework/value_t.h"                                // CoorsT, ShapeT
#include "gqten/framework/bases/streamable.h"                       // Streamable
#include "gqten/gqtensor/index.h"                                   // IndexVec, GetQNSctNumOfIdxs, CalcDiv
#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"     // BlockSparseDataTensor
#include "gqten/utility/utils_inl.h"                                // GenAllCoors, Rand, Reorder, CalcScalarNorm, CalcConj

#include <vector>       // vector
#include <iostream>     // cout, endl
#include <iterator>     // next
#include <algorithm>    // is_sorted

#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


/**
Symmetry-blocked sparse tensor.

@tparam ElemT Type of the tensor element.
@tparam QNT   Type of the quantum number.
*/
template <typename ElemT, typename QNT>
//class GQTensor : public Streamable {
class GQTensor {
public:
  // Constructors and destructor.
  /// Default constructor.
  GQTensor(void) = default;
  GQTensor(const IndexVec<QNT> &);
  GQTensor(const GQTensor &);
  GQTensor &operator=(const GQTensor &);
  GQTensor(GQTensor &&) noexcept;
  GQTensor &operator=(GQTensor &&) noexcept;
  /// Destroy a GQTensor.
  ~GQTensor(void) { delete pblk_spar_data_ten_; }

  // Get or check basic properties of the GQTensor.
  /// Get rank of the GQTensor.
  size_t Rank(void) const { return rank_; }

  /// Get the size of the GQTensor.
  size_t size(void) const { return size_; }

  /// Get indexes of the GQTensor.
  const IndexVec<QNT> &GetIndexes(void) const { return indexes_; }

  /// Get the shape of the GQTensor.
  const ShapeT &GetShape(void) const { return shape_; }

  /// Get the number of quantum number block contained by this GQTensor.
  size_t GetQNBlkNum(void) const {
    if (IsScalar() || IsDefault()) { return 0; }
    return pblk_spar_data_ten_->GetBlkIdxDataBlkMap().size();
  }

  /// Get the block sparse data tensor constant.
  const BlockSparseDataTensor<ElemT, QNT> &GetBlkSparDataTen(void) const {
    return *pblk_spar_data_ten_;
  }

  /// Get the pointer which point to block sparse data tensor constant.
  const BlockSparseDataTensor<ElemT, QNT> *GetBlkSparDataTenPtr(void) const {
    return pblk_spar_data_ten_;
  }

  /// Check whether the tensor is a scalar.
  bool IsScalar(void) const { return (rank_ == 0) && (size_ == 1); }

  /// Check whether the tensor is created by the default constructor.
  bool IsDefault(void) const { return size_ == 0; }

  // Calculate properties of the GQTensor.
  QNT Div(void) const;

  // Element getter and setter.
  ElemT GetElem(const std::vector<size_t> &) const;
  void SetElem(const std::vector<size_t> &, const ElemT);
  struct GQTensorElementAccessDeref;
  GQTensorElementAccessDeref operator[](std::vector<size_t> coors) {
    return GQTensorElementAccessDeref(*this, coors);
  }

  // Inplace operations.
  void Random(const QNT &);
  void Transpose(const std::vector<size_t> &);
  GQTEN_Double Normalize(void);
  void Dag(void);

  // Operators overload.
  bool operator==(const GQTensor &) const;
  bool operator!=(const GQTensor &rhs) const { return !(*this == rhs); }

  GQTensor operator+(const GQTensor &) const;
  GQTensor &operator+=(const GQTensor &);


private:
  /// The rank of the GQTensor.
  size_t rank_ = 0;
  /// The shape of the GQTensor.
  ShapeT shape_;
  /// The total number of elements of the GQTensor.
  size_t size_ = 0;

  /// Indexes of the GQTensor.
  IndexVec<QNT> indexes_;
  /// The pointer which point to block sparse data tensor.
  BlockSparseDataTensor<ElemT, QNT> *pblk_spar_data_ten_ = nullptr;

  /// The value of the rank 0 tensor (scalar).
  ElemT scalar_ = 0.0;

  ShapeT CalcShape_(void) const {
    ShapeT shape(indexes_.size());
    for (size_t i = 0; i < rank_; ++i) {
      shape[i] = indexes_[i].dim();
    }
    return shape;
  }

  size_t CalcSize_(void) const {
    size_t size = 1;
    for (auto dim : shape_) { size *= dim; }
    return size;
  }

  std::pair<CoorsT, CoorsT> CoorsToBlkCoorsDataCoors_(
      const CoorsT &coors
  ) const {
    assert(coors.size() == rank_);
    CoorsT blk_coors, data_coors;
    for (size_t i = 0; i < coors.size(); ++i) {
      auto blk_coor_data_coor = indexes_[i].CoorToBlkCoorDataCoor(coors[i]);
      blk_coors.push_back(blk_coor_data_coor.first);
      data_coors.push_back(blk_coor_data_coor.second);
    }
    return make_pair(blk_coors, data_coors);
  }
};


/**
Create an empty GQTensor using indexes.

@param indexes Vector of Index of the tensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(
    const IndexVec<QNT> &indexes
) : indexes_(indexes) {
  rank_ = indexes_.size();
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (!IsScalar()) {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  }
}


/**
Copy a GQTensor.

@param gqten Another GQTensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(const GQTensor &gqten) :
    rank_(gqten.rank_),
    shape_(gqten.shape_),
    size_(gqten.size_),
    indexes_(gqten.indexes_) {
  if (gqten.IsDefault()) {
    // Do nothing
  } else if (gqten.IsScalar()) {
    scalar_ = gqten.scalar_;
  } else {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(
                              *gqten.pblk_spar_data_ten_
                          );
    pblk_spar_data_ten_->pgqten_indexes = &indexes_;
  }
}


/**
Assign a GQTensor.

@param rhs Another GQTensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator=(const GQTensor &rhs) {
  rank_ = rhs.rank_;
  shape_ = rhs.shape_;
  size_ = rhs.size_;
  indexes_ = rhs.indexes_;
  if (rhs.IsDefault()) {
    // Do nothing
  } else if (rhs.IsScalar()) {
    scalar_ = rhs.scalar_;
  } else {
    delete pblk_spar_data_ten_;
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(
                              *rhs.pblk_spar_data_ten_
                          );
    pblk_spar_data_ten_->pgqten_indexes = &indexes_;
  }
  return *this;
}


/**
Move a GQTensor.

@param gqten Another GQTensor to-be moved.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(GQTensor &&gqten) noexcept :
    rank_(gqten.rank_),
    shape_(gqten.shape_),
    size_(gqten.size_) {
  if (gqten.IsDefault()) {
    // Do nothing
  } else if (gqten.IsScalar()) {
    scalar_ = gqten.scalar_;
  } else {
    indexes_ = std::move(gqten.indexes_);
    pblk_spar_data_ten_ = gqten.pblk_spar_data_ten_;
    gqten.pblk_spar_data_ten_ = nullptr;
  }
}


/**
Move and assign a GQTensor.

@param rhs Another GQTensor to-be moved.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator=(GQTensor &&rhs) noexcept {
  rank_ = rhs.rank_;
  shape_ = rhs.shape_;
  size_ = rhs.size_;
  if (rhs.IsDefault()) {
    // Do nothing
  } else if (rhs.IsScalar()) {
    scalar_ = rhs.scalar_;
  } else {
    indexes_ = std::move(rhs.indexes_);
    delete pblk_spar_data_ten_;
    pblk_spar_data_ten_ = rhs.pblk_spar_data_ten_;
    rhs.pblk_spar_data_ten_ = nullptr;
  }
  return *this;
}


template <typename ElemT, typename QNT>
struct GQTensor<ElemT, QNT>:: GQTensorElementAccessDeref {
  GQTensor &this_ten;
  std::vector<size_t> coors;

  GQTensorElementAccessDeref(
      GQTensor &gqten,
      const std::vector<size_t> &coors
  ) : this_ten(gqten), coors(coors) {}

  operator ElemT() const {
    return this_ten.GetElem(coors);
  }

  ElemT &operator=(const ElemT elem) {
    this_ten.SetElem(coors, elem);
  }
};


/**
Calculate the quantum number divergence of the GQTensor.

@return The quantum number divergence.
*/
template <typename ElemT, typename QNT>
QNT GQTensor<ElemT, QNT>::Div(void) const {
  if (IsScalar()) {
    std::cout << "Tensor is rank 0 (a scalar). Return QN()." << std::endl;
    return QNT();
  }
  auto blk_idx_data_blk_map = pblk_spar_data_ten_->GetBlkIdxDataBlkMap();
  auto blk_num = blk_idx_data_blk_map.size();
  if (blk_num == 0) {
    std::cout << "Tensor does not have a block. Return QN()." << std::endl;
    return QNT();
  }
  auto beg_it = blk_idx_data_blk_map.begin();
  auto div = CalcDiv(indexes_, beg_it->second.blk_coors);
  for (auto it = std::next(beg_it); it != blk_idx_data_blk_map.end(); ++it) {
    auto blk_i_div = CalcDiv(indexes_, it->second.blk_coors);
    if (blk_i_div != div) {
      std::cout << "Tensor does not have a special divergence. Return QN()."
                << std::endl;
      return QNT();
    }
  }
  return div;
}


/**
Get the tensor element using its coordinates.

@coors The coordinates of the tensor element. An empty vector for scalar.
*/
template <typename ElemT, typename QNT>
ElemT GQTensor<ElemT, QNT>::GetElem(const std::vector<size_t> &coors) const {
  assert(coors.size() == rank_);
  if (IsScalar()) { return scalar_; }
  auto blk_coors_data_coors = CoorsToBlkCoorsDataCoors_(coors);
  return pblk_spar_data_ten_->ElemGet(blk_coors_data_coors);
}


/**
Set the tensor element using its coordinates.

@param coors The coordinates of the tensor element. An empty vector for scalar.
@param elem The value of the tensor element.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::SetElem(
    const std::vector<size_t> &coors,
    const ElemT elem
) {
  assert(coors.size() == rank_);
  if (IsScalar()) {
    scalar_ = elem;
  } else {
    auto blk_coors_data_coors = CoorsToBlkCoorsDataCoors_(coors);
    pblk_spar_data_ten_->ElemSet(blk_coors_data_coors, elem);
  }
}


/**
Equivalence check.

@param rhs The GQTensor at the right hand side.

@return Equivalence check result.
*/
template <typename ElemT, typename QNT>
bool GQTensor<ElemT, QNT>::operator==(const GQTensor &rhs) const {
  // Default check
  if (IsDefault()) {
    if (rhs.IsDefault()) {
      return true;
    } else {
      return false;
    }
  }
  // Scalar check
  if (IsScalar() && rhs.IsScalar() && scalar_ != rhs.scalar_) { return false; }
  if (IsScalar()) {
    if (rhs.IsScalar() && scalar_ == rhs.scalar_) {
      return true;
    } else {
      return false;
    }
  }
  // Indexes check
  if (indexes_ != rhs.indexes_) { return false; }
  // Block sparse data tensor check
  return (*pblk_spar_data_ten_ == *rhs.pblk_spar_data_ten_);
}


/**
Random set tensor elements in [0, 1] with given quantum number divergence.
Original data of this tensor will be destroyed.

@param div Target quantum number divergence.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Random(const QNT &div) {
  assert(!IsDefault());
  if (IsScalar()) {
    assert(div == QNT());
    Rand(scalar_);
    return;
  }
  pblk_spar_data_ten_->Clear();
  for (auto &blk_coors : GenAllCoors(pblk_spar_data_ten_->blk_shape)) {
    if (CalcDiv(indexes_, blk_coors) == div) {
      pblk_spar_data_ten_->DataBlkInsert(blk_coors, false);     // NO allocate memory on this stage.
    }
  }
  pblk_spar_data_ten_->Random();
}


/**
Transpose the tensor using a new indexes order.

@param transed_idxes_order Transposed order of indexes.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Transpose(
    const std::vector<size_t> &transed_idxes_order
) {
  if (IsDefault() || IsScalar()) { return; }

  assert(transed_idxes_order.size() == rank_);
  // Give a shorted order, do nothing
  if (std::is_sorted(transed_idxes_order.begin(), transed_idxes_order.end())) {
    return;
  }
  Reorder(shape_, transed_idxes_order);
  Reorder(indexes_, transed_idxes_order);
  pblk_spar_data_ten_->Transpose(transed_idxes_order);
}


/**
Normalize the tensor and return its norm.

@return The norm before the normalization.
*/
template <typename ElemT, typename QNT>
GQTEN_Double GQTensor<ElemT, QNT>::Normalize(void) {
  if (IsDefault()) {
    std::cout << "Default GQTensor cannot be normailzed!" << std::endl;
    exit(1);
  } else if (IsScalar()) {
    GQTEN_Double norm = CalcScalarNorm(scalar_);
    scalar_ /= norm;
    return norm;
  } else {
    GQTEN_Double norm = pblk_spar_data_ten_->Normalize();
    return norm;
  }
}


/**
Switch the direction of the indexes, complex conjugate of the elements.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Dag(void) {
  if (IsDefault()) {
    std::cout << "Default GQTensor cannot be daggered!" << std::endl;
    exit(1);
  } else if (IsScalar()) {
    scalar_ = CalcConj(scalar_);
  } else {
    for (auto &index : indexes_) { index.Inverse(); }
    pblk_spar_data_ten_->Conj();
  }
}


/**
Add this GQTensor \f$ A \f$ and another GQTensor \f$ B \f$.

@param rhs Another GQTensor \f$ B \f$.

@return \f$ A + B \f$.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> GQTensor<ElemT, QNT>::operator+(
    const GQTensor &rhs
) const {
  assert(!(IsDefault() || rhs.IsDefault()));
  if (IsScalar()) {
    assert(rhs.IsScalar());
    GQTensor<ElemT, QNT> res(IndexVec<QNT>{});
    res.scalar_ = scalar_ + rhs.scalar_;
    return res;
  } else {
    assert(indexes_ == rhs.indexes_);
    GQTensor<ElemT, QNT> res(indexes_);
    res.pblk_spar_data_ten_->AddTwoBSDTAndAssignIn(
        *pblk_spar_data_ten_,
        *rhs.pblk_spar_data_ten_
    );
    return res;
  }
}


/**
Add and assign another GQTensor \f$ B \f$ to this tensor.

@param rhs Another GQTensor \f$ B \f$.

@return \f$ A = A + B \f$.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator+=(const GQTensor &rhs) {
  assert(!(IsDefault() || rhs.IsDefault()));
  if (IsScalar()) {
    assert(rhs.IsScalar());
    scalar_ += rhs.scalar_;
    return *this;
  } else {
    assert(indexes_ == rhs.indexes_);
    pblk_spar_data_ten_->AddAndAssignIn(*rhs.pblk_spar_data_ten_);
    return *this;
  }
}


// Helper functions.

/**
Calculate dagger of a GQTensor.

@param t A GQTensor \f$ T \f$.

@return Daggered \f$ T^{\dagger} \f$.
*/
template <typename GQTensorT>
GQTensorT Dag(const GQTensorT &t) {
  GQTensorT t_dag(t);
  t_dag.Dag();
  return t_dag;
}

/**
Calculate the quantum number divergence of a GQTensor.

@param t A GQTensor.

@return The quantum number divergence.
*/
template <typename ElemT, typename QNT>
QNT Div(const GQTensor<ElemT, QNT> &t) {
  assert(!t.IsDefault());
  if (t.IsScalar()) {
    std::cout << "Tensor is a scalar. Return empty quantum number."
              << std::endl;
    return QNT();
  } else {
    auto qnblk_num = t.GetQNBlkNum();
    if (qnblk_num == 0) {
      std::cout << "Tensor does not have a block. Return empty quantum number."
                << std::endl;
      return QNT();
    } else {
      auto blk_idx_data_blk_map = t.GetBlkSparDataTen().GetBlkIdxDataBlkMap();
      auto indexes = t.GetIndexes();
      auto first_blk_idx_data_blk = blk_idx_data_blk_map.begin();
      auto div = CalcDiv(indexes, first_blk_idx_data_blk->second.blk_coors);
      for (
          auto it = std::next(first_blk_idx_data_blk);
          it != blk_idx_data_blk_map.end();
          ++it
      ) {
        auto blki_div = CalcDiv(indexes, it->second.blk_coors);
        if (blki_div != div) {
          std::cout << "Tensor does not have a special divergence. Return empty quantum number."
                    << std::endl;
          return QNT();
        }
      }
      return div;
    }
  }
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_GQTENSOR_H */
