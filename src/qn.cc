// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-06 17:15
* 
* Description: GraceQ/tensor project. Implementation details about quantum number.
*/
#include "gqten/gqten.h"
#include "gqten/detail/vec_hash.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


QN::QN(void) { hash_ = CalcHash(); }


QN::QN(const std::vector<QNNameVal> &nm_vals) {
  for (auto &nm_val : nm_vals) {
    values_.push_back(nm_val.val);
  }
  hash_ = CalcHash();
}


QN::QN(const std::vector<long> &qn_vals) : values_(qn_vals) {
  hash_ = CalcHash();
}


QN::QN(const QN &qn) : values_(qn.values_), hash_(qn.hash_) {}


QN &QN::operator=(const QN &rhs) {
  values_ = rhs.values_;
  hash_ = rhs.hash_;
  return *this;
}


std::size_t QN::Hash(void) const { return hash_; }


std::size_t QN::CalcHash(void) const {
  if (values_.size() == 0) {
    return 0; 
  } else {
    return VecStdTypeHasher(values_);
  }
}


QN QN::operator-(void) const {
  auto qn_vals_size = this->values_.size();
  std::vector<long> new_qn_vals(qn_vals_size);
  for (std::size_t i = 0; i < qn_vals_size; ++i) {
    new_qn_vals[i] = - this->values_[i];
  }
  return QN(new_qn_vals);
}


QN &QN::operator+=(const QN &rhs) {
  auto qn_vals_size = this->values_.size();
  assert(qn_vals_size == rhs.values_.size());
  for (std::size_t i = 0; i < qn_vals_size; ++i) {
    this->values_[i] += rhs.values_[i];
  }
  hash_ = CalcHash();
  return *this;
}


QN operator+(const QN &lhs, const QN &rhs) {
  QN sum(lhs);
  sum += rhs;
  return sum;
}


QN operator-(const QN &lhs, const QN &rhs) {
  return lhs + (-rhs);
}


bool operator==(const QN &lhs, const QN &rhs) {
  return lhs.Hash() == rhs.Hash();
}


bool operator!=(const QN &lhs, const QN &rhs) {
  return !(lhs == rhs);
}


std::ifstream &bfread(std::ifstream &ifs, QN &qn) {
  long qn_vals_num;
  ifs >> qn_vals_num;
  qn.values_ = std::vector<long>(qn_vals_num);
  for (auto &value : qn.values_) { ifs >> value; }
  ifs >> qn.hash_;
  return ifs;
}


std::ofstream &bfwrite(std::ofstream &ofs, const QN &qn) {
  long qn_vals_num = qn.values_.size();
  ofs << qn_vals_num << std::endl;
  for (auto &value : qn.values_) { ofs << value << std::endl; }
  ofs << qn.hash_ << std::endl;
  return ofs;
}
} /* gqten */ 
