// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-06 17:39
* 
* Description: GraceQ/tensor project. Implementation details about quantum number sector.
*/
#include "gqten/gqten.h"
#include "vec_hash.h"

#include <vector>
#include <iostream>
#include <fstream>


namespace gqten {


QNSector &QNSector::operator=(const QNSector &rhs) {
  qn = rhs.qn;
  dim = rhs.dim;
  hash_ = rhs.hash_;
  return *this;
}


bool operator==(const QNSector &lhs, const QNSector &rhs) {
  return lhs.Hash() == rhs.Hash();
}


bool operator!=(const QNSector &lhs, const QNSector &rhs) {
  return !(lhs == rhs);
}


std::ifstream &bfread(std::ifstream &ifs, QNSector &qnsct) {
  bfread(ifs, qnsct.qn) >> qnsct.dim >> qnsct.hash_;
  return ifs;
}


std::ofstream &bfwrite(std::ofstream &ofs, const QNSector &qnsct) {
  bfwrite(ofs, qnsct.qn);
  ofs << qnsct.dim << std::endl;
  ofs << qnsct.hash_ << std::endl;
  return ofs;
}


QNSectorSet::QNSectorSet(const std::vector<const QNSector *> &pqnscts) {
  for (auto &pqnsct : pqnscts) { qnscts.push_back(*pqnsct); }
}


inline size_t QNSectorSet::Hash(void) const { return VecHasher(qnscts); }


bool operator==(const QNSectorSet &lhs, const QNSectorSet &rhs) {
  return lhs.Hash() == rhs.Hash();
}


bool operator!=(const QNSectorSet &lhs, const QNSectorSet &rhs) {
  return !(lhs == rhs);
}
} /* gqten */ 
