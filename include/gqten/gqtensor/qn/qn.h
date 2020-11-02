// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-30 15:07
*
* Description: GraceQ/tensor project. The quantum number of a system.
*/

/**
@file qn.h
@brief The quantum number of a system.
*/
#ifndef GQTEN_GQTENSOR_QN_QN_H
#define GQTEN_GQTENSOR_QN_QN_H


#include "gqten/gqtensor/qn/qnval.h"              // QNVal, QNValSharedPtr
#include "gqten/framework/bases/hashable.h"       // Hashable
#include "gqten/framework/bases/streamable.h"     // Streamable
#include "gqten/framework/vec_hash.h"             // VecPtrHasher

#include <assert.h>


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


/**
Quantum number card for name-value pair.
*/
class QNCard {
public:
  /**
  Create a quantum number card.

  @param name The name of the quantum number value. For example "Sz", "N", et. al.
  @param qnval The quantum number value.
  */
  QNCard(const std::string &name, const QNVal &qnval) :
      name_(name), pqnval_(qnval.Clone()) {}

  /// Get a std::shared_ptr which point to the quantum number value.
  QNValSharedPtr GetValPtr(void) const { return pqnval_; }

private:
  std::string name_;
  QNValSharedPtr pqnval_;
};

using QNCardVec = std::vector<QNCard>;


/**
The quantum number of a system.

@tparam QNValTs Types of quantum number values of the quantum number.
*/
template <typename... QNValTs>
class QN : public Hashable, public Streamable {
public:
  QN(void);
  QN(const QNCardVec &);
  ~QN(void);

  QN(const QN &);
  QN &operator=(const QN &);

  size_t dim(void) const { return dim_; }

  QN operator-(void) const;
  QN &operator+=(const QN &);

  QN operator+(const QN &rhs) const {
    QN sum(*this);
    sum += rhs;
    return sum;
  }
  QN operator-(const QN &rhs) const { return (*this) + (-rhs); }

  size_t Hash(void) const override { return hash_; }

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

private:
  QNValPtrVec pqnvals_;
  size_t dim_;
  size_t hash_;

  size_t CalcDim_(void) const;
  size_t CalcHash_(void) const;

  QN(const QNValPtrVec &);    // Intra used private constructor
};


template <typename... QNValTs>
inline size_t QN<QNValTs...>::CalcDim_(void) const {
  if (pqnvals_.empty()) {
    return 0;
  } else {
    size_t dim = 1;
    for (auto pqnval : pqnvals_) {
      dim *= pqnval->dim();
    }
    return dim;
  }
}


template <typename... QNValTs>
inline size_t QN<QNValTs...>::CalcHash_(void) const {
  if (pqnvals_.size() == 0) {
    return 0;
  } else {
    return VecPtrHasher(pqnvals_);
  }
}


/**
Create a default quantum number.
*/
template <typename... QNValTs>
QN<QNValTs...>::QN(void) : dim_(CalcDim_()), hash_(CalcHash_()) {}


/**
Create a quantum number using a vector of quantum number cards.

@param qncards A vector of quantum number cards.
*/
template <typename... QNValTs>
QN<QNValTs...>::QN(const QNCardVec &qncards) {
  for (auto &qncard : qncards) {
    pqnvals_.push_back((qncard.GetValPtr())->Clone());
  }
  dim_ = CalcDim_();
  hash_ = CalcHash_();
}


// WARNING: private constructor
template <typename... QNValTs>
QN<QNValTs...>::QN(const QNValPtrVec &pqnvals) {
  pqnvals_ = pqnvals;
  dim_ = CalcDim_();
  hash_ = CalcHash_();
}


template <typename... QNValTs>
QN<QNValTs...>::~QN(void) {
  for (auto &qnval : pqnvals_) {
    delete qnval;
  }
}


/**
Copy from a quantum number instance.

@param qn A quantum number instance to be copied.
*/
template <typename... QNValTs>
QN<QNValTs...>::QN(const QN &qn) :
    dim_(qn.dim_), hash_(qn.hash_) {
  for (auto &qnval : qn.pqnvals_) {
    pqnvals_.push_back(qnval->Clone());
  }
}


/**
Assign from another quantum number instance.

@param rhs A quantum number instance.
*/
template <typename... QNValTs>
QN<QNValTs...> &QN<QNValTs...>::operator=(const QN &rhs) {
  for (auto &qnval : pqnvals_) {
    delete qnval;
  }
  pqnvals_.clear();
  for (auto &qnval : rhs.pqnvals_) {
    pqnvals_.push_back(qnval->Clone());
  }
  dim_ = rhs.dim_;
  hash_ = rhs.hash_;
  return *this;
}


/**
Calculate the negation of a quantum number.
*/
template <typename... QNValTs>
QN<QNValTs...> QN<QNValTs...>::operator-(void) const {
  QNValPtrVec new_qnvals;
  for (auto &qnval : pqnvals_) {
    new_qnvals.push_back(qnval->Minus());
  }
  return QN(new_qnvals);    // WARNING: use private constructor here
}


/**
Add and assign operation to a quantum number.
*/
template <typename... QNValTs>
QN<QNValTs...> &QN<QNValTs...>::operator+=(const QN &rhs) {
  auto qnvals_size = this->pqnvals_.size();
  assert(qnvals_size == rhs.pqnvals_.size());
  for (size_t i = 0; i < qnvals_size; ++i) {
    (this->pqnvals_[i])->AddAssign(rhs.pqnvals_[i]);
  }
  dim_ = CalcDim_();
  hash_ = CalcHash_();
  return *this;
}


template <typename... QNValTs>
void QN<QNValTs...>::StreamRead(std::istream &is) {
  pqnvals_ = {(new QNValTs)...};     // Initialize the quantum number value slots
  for (auto &qnval : pqnvals_) {
    is >> (*qnval);
  }
  is >> hash_;
  dim_ = CalcDim_();    // Recalculate the dimension.
}


template <typename... QNValTs>
void QN<QNValTs...>::StreamWrite(std::ostream &os) const {
  for (auto &qnval : pqnvals_) {
    os << (*qnval) << std::endl;
  }
  os << hash_ << std::endl;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_QN_QN_H */
