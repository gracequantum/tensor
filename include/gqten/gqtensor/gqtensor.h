// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-13 16:18
*
* Description: GraceQ/tensor project. Symmetry-blocked sparse tensor class and
* some short inline helper functions.
*/

/**
@file gqtensor.h
@brief Symmetry-blocked sparse tensor class and some short inline helper functions.
*/
#ifndef GQTEN_GQTENSOR_GQTENSOR_H
#define GQTEN_GQTENSOR_GQTENSOR_H


#include "gqten/framework/value_t.h"                                // CoorsT, ShapeT, GQTEN_Double, GQTEN_Complex
#include "gqten/framework/bases/streamable.h"                       // Streamable
#include "gqten/gqtensor/index.h"                                   // IndexVec
#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"     // BlockSparseDataTensor

#include <vector>       // vector
#include <iostream>     // istream, ostream


namespace gqten {


/**
Symmetry-blocked sparse tensor.

@tparam ElemT Type of the tensor element.
@tparam QNT   Type of the quantum number.
*/
template <typename ElemT, typename QNT>
class GQTensor : public Streamable {
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
  BlockSparseDataTensor<ElemT, QNT> &GetBlkSparDataTen(void) {
    return *pblk_spar_data_ten_;
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

  GQTensor operator-(void) const;
  GQTensor operator+(const GQTensor &) const;
  GQTensor &operator+=(const GQTensor &);
  GQTensor operator*(const ElemT) const;
  GQTensor& operator*=(const ElemT);

  // Override base class
  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;


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

  ShapeT CalcShape_(void) const;

  size_t CalcSize_(void) const;

  std::pair<CoorsT, CoorsT> CoorsToBlkCoorsDataCoors_(const CoorsT &) const;
};

// Out-of-class declaration and definition.
template <typename GQTensorT>
GQTensorT Dag(const GQTensorT &);

template <typename ElemT, typename QNT>
QNT Div(const GQTensor<ElemT, QNT> &);

template <typename QNT>
GQTensor<GQTEN_Complex, QNT> ToComplex(const GQTensor<GQTEN_Double, QNT> &);

template <typename ElemT, typename QNT>
inline GQTensor<ElemT, QNT> operator*(
    const GQTEN_Double scalar,
    const GQTensor<ElemT, QNT> &t
) {
  return t * scalar;
}

template <typename ElemT, typename QNT>
inline GQTensor<ElemT, QNT> operator*(
    const GQTEN_Complex scalar,
    const GQTensor<ElemT, QNT> &t
) {
  return t * scalar;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_GQTENSOR_H */
