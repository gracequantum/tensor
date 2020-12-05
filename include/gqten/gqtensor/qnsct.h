// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-01 11:40
*
* Description: GraceQ/tensor project. Quantum number sector which represent a
* degenerate linear space labeled by a quantum number.
*/

/**
@file qnsct.h
@brief Quantum number sector which represent a degenerate linear space labeled
       by a quantum number.
*/
#ifndef GQTEN_GQTENSOR_QNSCT_H
#define GQTEN_GQTENSOR_QNSCT_H


#include "gqten/framework/bases/hashable.h"       // Hashable
#include "gqten/framework/bases/streamable.h"     // Streamable

#include <vector>     // vector

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


/**
Quantum number sector which represent a degenerate linear space labeled by a
quantum number.

@tparam QNT Type of the quantum number.
*/
template <typename QNT>
class QNSector : public Hashable, public Streamable {
public:
  /**
  Create a quantum number sector using a quantum number and the degeneracy.
  @param qn The quantum number which label this quantum number sector.
  @param dgnc The degeneracy of the linear space.
  */
  QNSector(const QNT &qn, const size_t dgnc) : qn_(qn), dgnc_(dgnc) {
    dim_ = qn.dim() * dgnc_;
    hash_ = CalcHash_();
  }

  /**
  Create a default quantum number sector.
  */
  QNSector(void) : QNSector(QNT(), 0) {}

  /**
  Copy a quantum number sector.
  */
  QNSector(const QNSector &qnsct) :
      qn_(qnsct.qn_),
      dgnc_(qnsct.dgnc_),
      dim_(qnsct.dim_),
      hash_(qnsct.hash_) {}

  /**
  Assign a quantum number sector using another one.
  */
  QNSector &operator=(const QNSector &rhs) {
    qn_ = rhs.qn_;
    dgnc_ = rhs.dgnc_;
    dim_ = rhs.dim_;
    hash_ = rhs.hash_;
    return *this;
  }

  /**
  Get the quantum number of this quantum number sector.
  */
  QNT GetQn(void) const { return qn_; }

  /**
  Get the degeneracy of this quantum number sector.
  */
  size_t GetDegeneracy(void) const { return dgnc_; }

  /**
  Get the dimension of this quantum number sector. The dimension of a quantum
  number sector is defined as the product of the dimension of the quantum number
  and the degeneracy of the quantum number sector, and equals the dimension of
  the linear space represented.
  */
  size_t dim(void) const { return dim_; }

  /**
  Calculate the coordinate in the degeneracy space (data coordinate) from the
  actual coordinate in the linear space represented by this quantum number
  sector.

  @param coor Coordinate in the linear space.

  @return Coordinate in the degeneracy spaece.
  */
  size_t CoorToDataCoor(const size_t coor) const {
    assert(coor < dim_);
    return coor % dgnc_;
  }

  size_t Hash(void) const override { return hash_; }

  void StreamRead(std::istream &is) override {
    is >> qn_ >> dgnc_ >> hash_;
    dim_ = qn_.dim() * dgnc_;
  }

  void StreamWrite(std::ostream &os) const override {
    os << qn_;
    os << dgnc_ << std::endl;
    os << hash_ << std::endl;
  }

private:
  QNT qn_;
  size_t dgnc_;
  size_t dim_;
  size_t hash_;

  size_t CalcHash_(void) const { return qn_.Hash() ^ dgnc_; }
};


template <typename QNT>
using QNSectorVec = std::vector<QNSector<QNT>>;
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_QNSCT_H */
