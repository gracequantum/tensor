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


#include "gqten/framework/value_t.h"            // CoorsT
#include "gqten/framework/bases/hashable.h"     // Hashable
#include "gqten/framework/bases/streamable.h"   // Streamable
#include "gqten/framework/vec_hash.h"           // VecHasher
#include "gqten/gqtensor/qnsct.h"               // QNSectorVec

#include <functional>     // std::hash

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


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
  GQTenIndexDirType GetDir(void) const { return dir_; }

  /**
  Get the number of qnsectors of the Index.
  */
  size_t GetQNSctNum(void) const { return qnscts_.size(); }

  /**
  Get a quantum number sector using sector coordinate.

  @param sct_coor The sector coordinate of the quantum number sector.
  */
  const QNSector<QNT> &GetQNSct(const size_t sct_coor) const {
    return qnscts_[sct_coor];
  }

  /**
  Get a quantum number sector using actual coordinate.

  @param actual_coor The actual coordinate.
  */
  const QNSector<QNT> &GetQNSctFromActualCoor(const size_t actual_coor) {
    auto sct_coor_data_coor = CoorToBlkCoorDataCoor(actual_coor);
    return qnscts_[sct_coor_data_coor.first];
  }

  /**
  Calculate block coordinate and data coordinate from corresponding actual coordinate.

  @param coor The actual coordinate.

  @return Block coordinate and data coordinate pair.
  */
  std::pair<size_t, size_t> CoorToBlkCoorDataCoor(const size_t coor) const {
    assert(coor < dim_);
    size_t residue_coor = coor;
    size_t blk_coor, data_coor;
    for (size_t i = 0; i < qnscts_.size(); ++i) {
      auto qnsct_dim = qnscts_[i].dim();
      if (residue_coor < qnsct_dim) {
        blk_coor = i;
        data_coor = qnscts_[i].CoorToDataCoor(residue_coor);
        break;
      } else {
        residue_coor -= qnsct_dim;
      }
    }
    return std::make_pair(blk_coor, data_coor);
  }

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


template <typename QNT>
using IndexVec = std::vector<Index<QNT>>;


/**
Calculate the number of quantum number sectors of Index from a vector of Index.

@param indexes A vector of Index.
*/
template <typename IndexVecT>
std::vector<size_t> CalcQNSctNumOfIdxs(const IndexVecT &indexes) {
  std::vector<size_t> qnsct_num_of_idxes;
  for (auto &index : indexes) {
    qnsct_num_of_idxes.push_back(index.GetQNSctNum());
  }
  return qnsct_num_of_idxes;
}


/**
Calculate quantum number divergence for a vector of Index and a given block coordinates.
*/
template <typename QNT>
QNT CalcDiv(const IndexVec<QNT> &indexes, const CoorsT &blk_coors) {
  assert(indexes.size() == blk_coors.size());
  QNT div, qnflux;
  auto ndim = indexes.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto index = indexes[i];
    if (index.GetDir() == GQTenIndexDirType::IN) {
      qnflux = -index.GetQNSct(blk_coors[i]).GetQn();
    } else if (index.GetDir() == GQTenIndexDirType::OUT) {
      qnflux = index.GetQNSct(blk_coors[i]).GetQn();
    }
    if (ndim == 1) {
      return qnflux;
    } else {
      if (i == 0) {
        div = qnflux;
      } else {
        div += qnflux;
      }
    }
  }
  return div;
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_INDEX_H */
