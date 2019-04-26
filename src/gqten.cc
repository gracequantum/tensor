/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:38
* 
* Description: GraceQ/tensor project. The main source code file.
*/
#include "gqten/gqten.h"
#include "hasher.h"

#include <vector>

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqten {


QN::QN(QNNameValIniter nm_vals) {
  for (auto &nm_val : nm_vals) {
    names_.push_back(nm_val.name);
    values_.push_back(nm_val.val);
  }
}


QN::QN(const std::vector<QNNameVal> &nm_vals) {
  for (auto &nm_val : nm_vals) {
    names_.push_back(nm_val.name);
    values_.push_back(nm_val.val);
  }

}


QN::QN(const QN &qn) {
  names_ = qn.names_;
  values_ = qn.values_;
}


std::size_t QN::hash(void) const {
  if (names_.size() == 0) {
    return 0; 
  } else {
    std::size_t hash_val = 0;
    std::hash<std::string> hasher;
    for (int i = 0; i < names_.size(); i++) {
      hash_val ^= hasher(names_[i] + std::to_string(values_[i]));
    }
    return hash_val;
  }
}


// Overload unary minus operator.
QN QN::operator-(void) const {
  auto nm_vals_size = this->names_.size();
  std::vector<QNNameVal> new_nm_vals(nm_vals_size);
  for (size_t i = 0; i < nm_vals_size; i++) {
    new_nm_vals[i] = QNNameVal(this->names_[i], -this->values_[i]);
  }
  return QN(new_nm_vals);
}


QN &QN::operator+=(const QN &rhs) {
  assert(this->names_.size() == rhs.names_.size());
  auto nm_vals_size = this->names_.size();
  for (size_t i = 0; i < nm_vals_size; i++) {
    assert(this->names_[i] == rhs.names_[i]);
    this->values_[i] += rhs.values_[i];
  }
  return *this;
}


bool operator==(const QN &lhs, const QN &rhs) {
  return lhs.hash() == rhs.hash();
}


bool operator!=(const QN &lhs, const QN &rhs) {
  return !(lhs == rhs);
}


QN operator+(const QN &lhs, const QN &rhs) {
  QN sum = lhs;
  sum += rhs;
  return sum;
}


QN operator-(const QN &lhs, const QN &rhs) {
  return lhs + (-rhs);
}
} /* gqten */ 
