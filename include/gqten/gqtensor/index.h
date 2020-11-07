// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-02 11:28
*
* Description: GraceQ/tensor project. Linear space constructed by a series of
* QNSector, and it can define a direction.
*/

/**
@file index.h
@brief Linear space constructed by a series of QNSector, and it can define a
       direction.
*/
#ifndef GQTEN_GQTENSOR_INDEX_H
#define GQTEN_GQTENSOR_INDEX_H


#include "gqten/framework/bases/hashable.h"     // Hashable
#include "gqten/framework/bases/streamable.h"   // Streamable
#include "gqten/framework/vec_hash.h"           // VecHasher
#include "gqten/gqtensor/qnsct.h"               // QNSectorVec

#include <functional>     // std::hash


namespace gqten {


/// Possible directions for an index.
enum GQTenIndexDirType {
  NDIR = 0,      // Direction non-defined.
  IN =  -1,      // In direction.
  OUT =  1       // OUT direction.
};


/**
Linear space constructed by a series of QNSector, and it can define a direction.

@tparam QNT Type of the quantum number.
*/
template <typename QNT>
class Index : public Hashable, public Streamable {
public:
  /**
  Create an Index using a series of quantum number sectors and the direction.

  @param qnscts A series of quantum number sectors.
  @param dir The direction of this Index.
  */
  Index(const QNSectorVec<QNT> &qnscts, const GQTenIndexDirType dir) :
      qnscts_(qnscts), dir_(dir) {
    dim_ = CalcDim_();
    hash_ = CalcHash_();
  }

  /**
  Create a default Index.
  */
  Index(void) : Index({}, GQTenIndexDirType::NDIR) {}

  /**
  Copy an Index.

  @param index Another Index object.
  */
  Index(const Index &index) :
      qnscts_(index.qnscts_),
      dir_(index.dir_),
      dim_(index.dim_),
      hash_(index.hash_) {}

  /**
  Assign from another Index.

  @param rhs Another Index object.
  */
  Index &operator=(const Index &rhs) {
    qnscts_ = rhs.qnscts_;
    dir_ = rhs.dir_;
    dim_ = rhs.dim_;
    hash_ = rhs.hash_;
    return *this;
  }

  /**
  Get the dimension of the Index.
  */
  size_t dim(void) const { return dim_; }

  /**
  Get the direction of the Index.
  */
  GQTenIndexDirType GetDir(void) { return dir_; }

  /**
  Inverse the direction of the Index.
  */
  void Inverse(void) {
    switch (dir_) {
      case GQTenIndexDirType::IN:
        dir_ = GQTenIndexDirType::OUT;
        break;
      case GQTenIndexDirType::OUT:
        dir_ = GQTenIndexDirType::IN;
        break;
      case GQTenIndexDirType::NDIR:
        break;
      default:
        std::cout << "Invalid Index direction!" << std::endl;
        exit(1);
    }
    hash_ = CalcHash_();      // Recalculate hash value.
  }

  size_t Hash(void) const override { return hash_; }

  void StreamRead(std::istream &is) override {
    size_t qnscts_size;
    is >> qnscts_size;
    qnscts_ = QNSectorVec<QNT>(qnscts_size);
    for (auto &qnsct : qnscts_) { is >> qnsct; }
    int dir_int_repr;
    is >> dir_int_repr;
    dir_ = static_cast<GQTenIndexDirType>(dir_int_repr);
    is >> dim_;
    is >> hash_;
  }

  void StreamWrite(std::ostream &os) const override {
    os << qnscts_.size() << std::endl;
    for (auto &qnsct : qnscts_) { os << qnsct; }
    int dir_int_repr = dir_;
    os << dir_int_repr << std::endl;
    os << dim_ << std::endl;
    os << hash_ << std::endl;
  }


private:
  QNSectorVec<QNT> qnscts_;
  GQTenIndexDirType dir_;
  size_t dim_;
  size_t hash_;

  size_t CalcDim_(void) {
    size_t dim = 0;
    for (auto &qnsct : qnscts_) {
      dim += qnsct.dim();
    }
    return dim;
  }

  size_t CalcHash_(void) {
    std::hash<int> int_hasher;
    return VecHasher(qnscts_) ^ int_hasher(dir_);
  }
};


/**
Inverse an Index.

@tparam IndexT The type of the index.

@param idx A to-be inversed index.

@return The inversed index.
*/
template <typename IndexT>
IndexT InverseIndex(const IndexT &idx) {
  IndexT inv_idx(idx);
  inv_idx.Inverse();
  return inv_idx;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_INDEX_H */
