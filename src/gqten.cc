/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:38
* 
* Description: GraceQ/tensor project. The main source code file.
*/
#include "gqten/gqten.h"
#include "vec_hash.h"

#include <vector>
#include <string>

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqten {


// Quantum number.
//QN::QN(QNNameValIniter nm_vals) {
  //for (auto &nm_val : nm_vals) {
    //names_.push_back(nm_val.name);
    //values_.push_back(nm_val.val);
  //}
//}


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


std::size_t QN::Hash(void) const {
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
  return lhs.Hash() == rhs.Hash();
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


// Quantum number sector.
size_t QNSector::Hash(void) const {
  return qn.Hash() ^ int_hasher_(dim);
}


bool operator==(const QNSector &lhs, const QNSector &rhs) {
  return lhs.Hash() == rhs.Hash();
}


bool operator!=(const QNSector &lhs, const QNSector &rhs) {
  return !(lhs == rhs);
}


// Quantum number sector set.
size_t QNSectorSet::Hash(void) const {
  return VecHasher(qnscts);
}


bool operator==(const QNSectorSet &lhs, const QNSectorSet &rhs) {
  return lhs.Hash() == rhs.Hash();
}


bool operator!=(const QNSectorSet &lhs, const QNSectorSet &rhs) {
  return !(lhs == rhs);
}


// Index.
size_t Index::Hash(void) const {
  return QNSectorSet::Hash() ^ str_hasher_(tag);
}


InterOffsetQnsct Index::CoorOffsetAndQnsct(long coor) const {
  long inter_offset = 0;
  for (auto &qnsct : qnscts) {
    long temp_inter_offset = inter_offset + qnsct.dim;
    if (temp_inter_offset > coor) {
      return InterOffsetQnsct(inter_offset, qnsct);
    } else if (temp_inter_offset <= coor) {
      inter_offset = temp_inter_offset;
    }
  }
}


// Dense block labeled by the quantum number.
//QNBlock::QNBlock(const std::initializer_list<QNSector> &init_qnscts) :
    //QNSectorSet(init_qnscts) {
  //ndim = qnscts.size(); 
  //for (auto &qnsct : qnscts) {
    //shape.push_back(qnsct.dim);
  //}
  //if (ndim != 0) {
    //size = 1;       // Initialize the block size.
    //for (long i = 0; i < ndim; ++i) {
      //size *= shape[i];
    //}
    //data_ = new double[size] ();    // Allocate memory and initialize to 0.
    //data_offsets_ = CalcDataOffsets(shape);
  //}
//}


QNBlock::QNBlock(const std::vector<QNSector> &init_qnscts) :
    QNSectorSet(init_qnscts) {
  ndim = qnscts.size(); 
  for (auto &qnsct : qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) {
      size *= shape[i];
    }
    data_ = new double[size] ();    // Allocate memory and initialize to 0.
    data_offsets_ = CalcDataOffsets(shape);
  }
}


QNBlock::QNBlock(const QNBlock &qnblk) :
    QNSectorSet(qnblk.qnscts), 
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_) {
  data_ = new double [size] ();
  std::memcpy(data_, qnblk.data_, size * sizeof(double));
}


QNBlock &QNBlock::operator=(const QNBlock &rhs) {
  // Copy data.
  auto new_data = new double [rhs.size];
  std::memcpy(new_data, rhs.data_, rhs.size * sizeof(double));
  delete [] data_;
  data_ = new_data;
  // Copy other members.
  ndim = rhs.ndim;
  shape = rhs.shape;
  size = rhs.size;
  data_offsets_ = rhs.data_offsets_;
  return *this;
}


QNBlock::~QNBlock(void) {
  delete [] data_;
  data_ = nullptr;
}


// Block element getter.
const double &
QNBlock::operator()(const std::vector<long> &coors) const {
  assert(coors.size() == ndim);
  long offset = 0;
  for (size_t i = 0; i < ndim; ++i) {
    offset += coors[i] * data_offsets_[i];
  }
  return *(data_+offset);
}


//const double &
//QNBlock::operator()(const std::initializer_list<long> &coors) const {
  //return QNBlock::operator()(std::vector<long>(coors));
//}


// Block element setter.
double &
QNBlock::operator()(const std::vector<long> &coors) {
  assert(coors.size() == ndim);
  auto offset = CalcOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


// Block element getter.
//double &
//QNBlock::operator()(const std::initializer_list<long> &coors) {
  //return QNBlock::operator()(std::vector<long>(coors));
//}


//size_t QNBlock::PartHash(const std::initializer_list<long> &dims) const {
size_t QNBlock::PartHash(const std::vector<long> &dims) const {
  std::vector<QNSector> selected_qnscts;
  for (auto &dim : dims) {
    selected_qnscts.push_back(qnscts[dim]);
  }
  return VecHasher(selected_qnscts);
}


// Inplace operation.
void QNBlock::Random(void) {
  for (int i = 0; i < size; ++i) {
    data_[i] = double(rand()) / RAND_MAX;
  }
}


void QNBlock::Transpose(const std::vector<long> &transed_axes) {
  std::vector<QNSector> transed_qnscts(ndim);
  std::vector<long> transed_shape(ndim);
  for (long i = 0; i < ndim; ++i) {
    transed_qnscts[i] = qnscts[transed_axes[i]];
    transed_shape[i] = transed_qnscts[i].dim;
  }
  auto transed_data_offsets_ = CalcDataOffsets(shape);
  TransposeBlkData(
      data_, ndim, transed_axes,
      shape, data_offsets_, transed_data_offsets_);
  shape = transed_shape;
  qnscts = transed_qnscts;
  data_offsets_ = transed_data_offsets_;
}


// Tensor with U1 symmetry.
GQTensor::GQTensor(const GQTensor &gqtensor) : indexes(gqtensor.indexes) {
  for (auto &blk : gqtensor.blocks_) {
    auto new_blk = new QNBlock(*blk);
    blocks_.push_back(new_blk);
  }
}


GQTensor &GQTensor::operator=(const GQTensor &rhs) {
  auto new_blk_num = rhs.blocks_.size();
  std::vector<QNBlock *> new_blks(new_blk_num);
  for (size_t i = 0; i < new_blk_num; ++i) {
    auto new_blk = new QNBlock(*rhs.blocks_[i]);
    blocks_[i] = new_blk;
  }
  indexes = rhs.indexes;
  return *this;
}


GQTensor::~GQTensor(void) {
  for (auto blk : blocks_) {
    delete blk;
  }
}


// GQTensor element getter.
//double GQTensor::Elem(const std::initializer_list<long> &coors) const {
double GQTensor::Elem(const std::vector<long> &coors) const {
  auto blk_coors_and_blk_key = TargetBlkCoorsAndBlkKey(coors);
  for (auto blk : blocks_) {
    if (blk->qnscts == blk_coors_and_blk_key.blk_key) {
      return (*blk)(blk_coors_and_blk_key.blk_coors);
    }
  }
  return 0.0;
}


//double GQTensor::Elem(const std::initializer_list<long> &coors) const {
  //return 
//}


// GQTensor element setter.
double &
//GQTensor::operator()(const std::initializer_list<long> &coors) {
GQTensor::operator()(const std::vector<long> &coors) {
  auto blk_coors_and_blk_key = TargetBlkCoorsAndBlkKey(coors);
  for (auto blk : blocks_) {
    if (blk->qnscts == blk_coors_and_blk_key.blk_key) {
      return (*blk)(blk_coors_and_blk_key.blk_coors);
    }
  }
  QNBlock *new_block = new QNBlock(blk_coors_and_blk_key.blk_key.qnscts);
  blocks_.push_back(new_block);
  return (*blocks_.back())(blk_coors_and_blk_key.blk_coors);
}


// Inplace operations.
// Tensor transpose.
//void GQTensor::Transpose(const std::initializer_list<long> &axes) {
void GQTensor::Transpose(const std::vector<long> &axes) {
  assert(axes.size() == indexes.size());
  // Transpose indexes.
  std::vector<long> transed_axes = axes;
  std::vector<Index> transed_indexes(indexes.size());
  for (size_t i = 0; i < transed_axes.size(); ++i) {
    transed_indexes[i] = indexes[transed_axes[i]];
  }
  indexes = transed_indexes;
  // Transpose blocks.
  for (auto &blk : blocks_) {
    blk->Transpose(transed_axes);
  }
}
// Random set tensor elements with given quantum number divergence.
void GQTensor::Random(const QN &div) {
  for (auto &blk_key : BlkKeysIter()) {
    if (CalcDiv(blk_key.qnscts, indexes) == div) {
      QNBlock *block = new QNBlock(blk_key.qnscts);
      block->Random();
      blocks_.push_back(block);
    }
  }
}


BlkCoorsAndBlkKey
GQTensor::TargetBlkCoorsAndBlkKey(const std::vector<long> &coors) const {
  std::vector<long> blk_coors(coors.size());
  std::vector<QNSector> blk_key(coors.size());
  for (size_t i = 0; i < coors.size(); ++i) {
    auto inter_offset_and_qnsct = indexes[i].CoorOffsetAndQnsct(coors[i]);
    blk_coors[i] = coors[i] - inter_offset_and_qnsct.inter_offset;
    blk_key[i] = inter_offset_and_qnsct.qnsct;
  }
  return BlkCoorsAndBlkKey(blk_coors, blk_key);
}


std::vector<QNSectorSet> GQTensor::BlkKeysIter(void) const {
  std::vector<std::vector<QNSector>> v;
  for (auto &index : indexes) {
    v.push_back(index.qnscts);
  }
  auto s = CalcCartProd(v);
  std::vector<QNSectorSet> blk_keys;
  for (auto &qnscts : s) {
    blk_keys.push_back(QNSectorSet(qnscts));
  }
  return blk_keys;
}


// GQTensor objects operations.
Index InverseIndex(const Index &idx) {
  Index inversed_idx = idx;
  if (idx.dir == IN) {
    inversed_idx.dir = OUT;
  } else if (idx.dir == OUT) {
    inversed_idx.dir = IN;
  }
  return inversed_idx;
}


// Helper functions.
QN CalcDiv(const QNSectorSet &blk_key, const std::vector<Index> &indexes) {
  QN div;
  auto ndim = indexes.size();
  assert(blk_key.qnscts.size() == ndim);
  for (size_t i = 0; i < ndim; ++i) {
    if (indexes[i].dir == IN) {
      auto qnflow = -blk_key.qnscts[i].qn;
      if (ndim == 1) {
        return qnflow;
      } else {
        if (i == 0) {
          div = qnflow;
        } else {
          div += qnflow;
        }
      }
    } else if (indexes[i].dir == OUT) {
      auto qnflow = blk_key.qnscts[i].qn;
      if (ndim == 1) {
        return qnflow;
      } else {
        if (i == 0) {
          div = qnflow;
        } else {
          div += qnflow;
        }
      }
    } 
  }
  return div;
}


std::vector<long> CalcDataOffsets(const std::vector<long> &shape) {
  auto ndim = shape.size();
  std::vector<long> offsets(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    if (i == ndim-1) {
      offsets[i] = 1;
    } else {
      offsets[i] = MulToEnd(shape, i+1);
    }
  }
  return offsets;
}


// Multiplication from vec[i] to the end.
long MulToEnd(const std::vector<long> &v, int i) {
  assert(i < v.size());
  long mul = 1;
  for (auto it = v.begin()+i; it != v.end(); ++it) {
    mul *= *it; 
  } 
  return mul;
}


void TransposeBlkData(
    double * &data, const long &ndim, const std::vector<long> &axes_map,
    const std::vector<long> &old_shape,
    const std::vector<long> &old_data_offsets,
    const std::vector<long> &new_data_offsets) {
  double *new_data = new double [ndim] ();
  for (auto &old_coors : GenAllCoors(old_shape)) {
    new_data[CalcOffset(TransCoors(old_coors, axes_map), ndim, new_data_offsets)] =
        data[CalcOffset(old_coors, ndim, old_data_offsets)];
  }
  data = new_data;
}


std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &shape) {
  std::vector<std::vector<long>> each_coors(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    for (long j = 0; j < shape[i]; ++j) {
      each_coors[i].push_back(j);
    }
  }
  return CalcCartProd(each_coors);
}


} /* gqten */ 
