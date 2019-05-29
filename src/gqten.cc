/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:38
* 
* Description: GraceQ/tensor project. The main source code file.
*/
#include "gqten/gqten.h"
#include "vec_hash.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>

#include "hptt.h"


namespace gqten {


QN::QN(void) { hash_ = CalcHash(); }


QN::QN(const std::vector<QNNameVal> &nm_vals) {
  for (auto &nm_val : nm_vals) {
    names_.push_back(nm_val.name);
    values_.push_back(nm_val.val);
  }
  hash_ = CalcHash();
}


QN::QN(const QN &qn) {
  names_ = qn.names_;
  values_ = qn.values_;
  hash_ = qn.hash_;
}


QN &QN::operator=(const QN &rhs) {
  names_ = rhs.names_;
  values_ = rhs.values_;
  hash_ = rhs.hash_;
  return *this;
}


std::size_t QN::Hash(void) const { return hash_; }


std::size_t QN::CalcHash(void) const {
  if (names_.size() == 0) {
    return 0; 
  } else {
    std::vector<HashableString> val_strs;
    for (auto &val : values_) { val_strs.push_back(HashableString(val)); }
    return VecHasher(val_strs);
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
  hash_ = CalcHash();
  return *this;
}


bool operator==(const QN &lhs, const QN &rhs) {
  return lhs.Hash() == rhs.Hash();
}


bool operator!=(const QN &lhs, const QN &rhs) {
  return !(lhs == rhs);
}


QN operator+(const QN &lhs, const QN &rhs) {
  QN sum(lhs);
  sum += rhs;
  return sum;
}


QN operator-(const QN &lhs, const QN &rhs) {
  return lhs + (-rhs);
}


std::ifstream &bfread(std::ifstream &ifs, QN &qn) {
  long nv_num;
  ifs >> nv_num;
  qn.names_ = std::vector<std::string>(nv_num);
  for (auto &name : qn.names_) { ifs >> name; }
  qn.values_ = std::vector<long>(nv_num);
  for (auto &value : qn.values_) { ifs >> value; }
  ifs >> qn.hash_;
  return ifs;
}


std::ofstream &bfwrite(std::ofstream &ofs, const QN &qn) {
  long nv_num = qn.names_.size();
  ofs << nv_num << std::endl;
  for (auto &name : qn.names_) { ofs << name << std::endl; }
  for (auto &value : qn.values_) { ofs << value << std::endl; }
  ofs << qn.hash_ << std::endl;
  return ofs;
}


// Quantum number sector.
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


// Quantum number sector set.
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


// Index.
std::hash<std::string> str_hasher_;


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


std::ifstream &bfread(std::ifstream &ifs, Index &idx) {
  long qnscts_num;
  ifs >> qnscts_num;
  idx.qnscts = std::vector<QNSector>(qnscts_num);
  for (auto &qnsct : idx.qnscts) { bfread(ifs, qnsct); }
  ifs >> idx.dim >> idx.dir;
  // Deal with empty tag, where will be '\n\n'.
  char next1_ch, next2_ch;
  ifs.get(next1_ch);
  ifs.get(next2_ch);
  if (next2_ch != '\n') {
    ifs.putback(next2_ch);
    ifs.putback(next1_ch);
    ifs >> idx.tag;
  }
  return ifs; 
}


std::ofstream &bfwrite(std::ofstream &ofs, const Index &idx) {
  long qnscts_num = idx.qnscts.size();
  ofs << qnscts_num << std::endl;
  for (auto &qnsct : idx.qnscts) { bfwrite(ofs, qnsct); }
  ofs << idx.dim << std::endl;
  ofs << idx.dir << std::endl;
  ofs << idx.tag << std::endl;
  return ofs;
}


// Dense block labeled by the quantum number.
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
    qnscts_hash_ = QNSectorSet::Hash();
  }
}


QNBlock::QNBlock(const std::vector<const QNSector *> &pinit_qnscts) :
    QNSectorSet(pinit_qnscts) {
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
    qnscts_hash_ = QNSectorSet::Hash();
  }
}


QNBlock::QNBlock(const QNBlock &qnblk) :
    QNSectorSet(qnblk),   // Use copy constructor of the base class.
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_) {
  data_ = new double[size] ();
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
  qnscts_hash_ = rhs.qnscts_hash_;
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
  auto offset = CalcOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


// Block element setter.
double &
QNBlock::operator()(const std::vector<long> &coors) {
  assert(coors.size() == ndim);
  auto offset = CalcOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


size_t QNBlock::PartHash(const std::vector<long> &dims) const {
  auto selected_qnscts_ndim  = dims.size();
  std::vector<const QNSector *> pselected_qnscts(selected_qnscts_ndim);
  for (std::size_t i = 0; i < selected_qnscts_ndim; ++i) {
    pselected_qnscts[i] = &qnscts[dims[i]];
  }
  return VecPtrHasher(pselected_qnscts);
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
  auto transed_data_offsets_ = CalcDataOffsets(transed_shape);
  auto new_data = TransposeData(
                      data_,
                      ndim, size, shape,
                      transed_axes);
  delete[] data_;
  data_ = new_data;
  shape = transed_shape;
  qnscts = transed_qnscts;
  data_offsets_ = transed_data_offsets_;
}


double *TransposeData(
    const double *old_data,
    const long &old_ndim,
    const long &old_size,
    const std::vector<long> &old_shape,
    const std::vector<long> &transed_axes) {
  int dim = old_ndim;
  int perm[dim];  for (int i = 0; i < dim; ++i) { perm[i] = transed_axes[i]; }
  int sizeA[dim]; for (int i = 0; i < dim; ++i) { sizeA[i] = old_shape[i]; }
  int outerSizeB[dim];
  for (int i = 0; i < dim; ++i) { outerSizeB[i] = old_shape[perm[i]]; }
  auto transed_data = new double[old_size];
  dTensorTranspose(perm, dim,
      1.0, old_data, sizeA, sizeA,
      0.0, transed_data, outerSizeB,
      1, 1);
  return transed_data;
}




std::ifstream &bfread(std::ifstream &ifs, QNBlock &qnblk) {
  ifs >> qnblk.ndim;

  ifs >> qnblk.size;

  qnblk.shape = std::vector<long>(qnblk.ndim);
  for (auto &order : qnblk.shape) { ifs >> order; }

  qnblk.qnscts = std::vector<QNSector>(qnblk.ndim);
  for (auto &qnsct : qnblk.qnscts) { bfread(ifs, qnsct); }

  qnblk.data_offsets_ = std::vector<long>(qnblk.ndim);
  for (auto &offset : qnblk.data_offsets_) { ifs >> offset; }

  ifs >> qnblk.qnscts_hash_;

  ifs.seekg(1, std::ios::cur);    // Skip the line break.

  if (qnblk.size != 0) {
    qnblk.data_ = new double[qnblk.size];
    ifs.read((char *) qnblk.data_, qnblk.size*sizeof(double));
  }

  return ifs;
}


std::ofstream &bfwrite(std::ofstream &ofs, const QNBlock &qnblk) {
  ofs << qnblk.ndim << std::endl;

  ofs << qnblk.size << std::endl;

  for (auto &order : qnblk.shape) { ofs << order << std::endl; }

  for (auto &qnsct : qnblk.qnscts) { bfwrite(ofs, qnsct); }

  for (auto &offset : qnblk.data_offsets_) { ofs << offset << std::endl; }

  ofs << qnblk.qnscts_hash_ << std::endl;

  if (qnblk.size != 0) {
    ofs.write((char *) qnblk.data_, qnblk.size*sizeof(double));
  }
  ofs << std::endl;

  return ofs;
}


// Tensor with U1 symmetry.
GQTensor::GQTensor(const std::vector<Index> &idxes) : indexes(idxes) {
  for (auto &index : indexes) {
    auto size = 0;
    for (auto &qnsct : index.qnscts) {
      size += qnsct.dim;
    }
    shape.push_back(size);
  }
}


GQTensor::GQTensor(const GQTensor &gqtensor) :
    indexes(gqtensor.indexes),
    scalar(gqtensor.scalar),
    shape(gqtensor.shape) {
  for (auto &blk : gqtensor.blocks_) {
    auto new_blk = new QNBlock(*blk);
    blocks_.push_back(new_blk);
  }
}


GQTensor &GQTensor::operator=(const GQTensor &rhs) {
  for (auto blk : blocks_) { delete blk; }
  auto new_blk_num = rhs.blocks_.size();
  std::vector<QNBlock *> new_blks(new_blk_num);
  for (size_t i = 0; i < new_blk_num; ++i) {
    auto new_blk = new QNBlock(*rhs.blocks_[i]);
    new_blks[i] = new_blk;
  }
  blocks_ = new_blks;
  scalar  = rhs.scalar;
  indexes = rhs.indexes;
  shape = rhs.shape;
  return *this;
}


GQTensor::~GQTensor(void) {
  for (auto blk : blocks_) { delete blk; }
}


// GQTensor element getter.
double GQTensor::Elem(const std::vector<long> &coors) const {
  auto blk_coors_and_blk_key = TargetBlkCoorsAndBlkKey(coors);
  for (auto blk : blocks_) {
    if (blk->qnscts == blk_coors_and_blk_key.blk_key) {
      return (*blk)(blk_coors_and_blk_key.blk_coors);
    }
  }
  return 0.0;
}


// GQTensor element setter.
double &
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
void GQTensor::Transpose(const std::vector<long> &axes) {
  assert(axes.size() == indexes.size());
  // Transpose indexes.
  std::vector<long> transed_axes = axes;
  std::vector<Index> transed_indexes(indexes.size());
  std::vector<long> transed_shape(shape.size());
  for (size_t i = 0; i < transed_axes.size(); ++i) {
    transed_indexes[i] = indexes[transed_axes[i]];
    transed_shape[i] = shape[transed_axes[i]];
  }
  indexes = transed_indexes;
  shape = transed_shape;
  // Transpose blocks.
  for (auto &blk : blocks_) {
    blk->Transpose(transed_axes);
  }
}


// Random set tensor elements with given quantum number divergence.
// Any original blocks will be destroyed.
void GQTensor::Random(const QN &div) {
  for (auto &blk : blocks_) { delete blk; }
  blocks_ = std::vector<QNBlock *>();
  for (auto &blk_key : BlkKeysIter()) {
    if (CalcDiv(blk_key.qnscts, indexes) == div) {
      QNBlock *block = new QNBlock(blk_key.qnscts);
      block->Random();
      blocks_.push_back(block);
    }
  }
}


// Normalize the GQTensor.
void GQTensor::Normalize(const double norm) {
  for (auto &blk : blocks_) {
    auto data = blk->DataRef();
    for (long i = 0; i < blk->size; ++i) {
      data[i] = data[i] / norm;
    }
  }
}


double GQTensor::Normalize(void) {
  auto norm = Norm();
  Normalize(norm);
  return norm;
}


// Operators Overload.
GQTensor GQTensor::operator+(const GQTensor &rhs) {
  auto added_t = GQTensor(indexes);
  for (auto &rhs_blk : rhs.BlksConstRef()) {
    auto  has_blk = false;
    for (auto &lhs_blk : blocks_) {
      if (lhs_blk->QNSectorSetHash() == rhs_blk->QNSectorSetHash()) {
        assert(lhs_blk->size == rhs_blk->size);
        auto added_blk = new QNBlock(lhs_blk->qnscts);
        auto added_data = new double [lhs_blk->size];
        auto lhs_blk_data = lhs_blk->DataConstRef();
        auto rhs_blk_data = rhs_blk->DataConstRef();
        for (long i = 0; i < lhs_blk->size; ++i) {
          added_data[i] = lhs_blk_data[i] + rhs_blk_data[i];
        }
        added_blk->DataRef() = added_data;
        added_t.BlksRef().push_back(added_blk);
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto added_blk = new QNBlock(*rhs_blk);
      added_t.BlksRef().push_back(added_blk);
    }
  }
  for (auto &lhs_blk : blocks_) {
    auto has_blk = false;
    for (auto &existed_blk : added_t.BlksConstRef()) {
      if (existed_blk->QNSectorSetHash() == lhs_blk->QNSectorSetHash()) {
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto added_blk = new QNBlock(*lhs_blk);
      added_t.BlksRef().push_back(added_blk);
    }
  }
  return added_t;
}


GQTensor &GQTensor::operator+=(const GQTensor &rhs) {
  if (this->indexes.size() == 0) {
    assert(this->indexes == rhs.indexes);
    this->scalar += rhs.scalar;
    return *this;
  }

  for (auto &prhs_blk : rhs.BlksConstRef()) {
    auto has_blk = false;
    for (auto &plhs_blk : blocks_) {
      if (plhs_blk->QNSectorSetHash() == prhs_blk->QNSectorSetHash()) {
        auto lhs_blk_data = plhs_blk->DataRef();
        auto rhs_blk_data = prhs_blk->DataConstRef();
        assert(plhs_blk->size == prhs_blk->size);
        for (long i = 0; i < prhs_blk->size; i++) {
          lhs_blk_data[i] += rhs_blk_data[i];
        }
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto pnew_blk = new QNBlock(*prhs_blk);
      blocks_.push_back(pnew_blk);
    }
  }
  return *this;
}


GQTensor GQTensor::operator-(void) const {
  auto minus_t = GQTensor(*this);
  for (auto &blk : minus_t.BlksRef()) {
    auto data = blk->DataRef();
    for (long i = 0; i < blk->size; ++i) {
      data[i] = -data[i];
    }
  }
  return minus_t;
}


GQTensor *GQTensor::operator-=(const GQTensor &rhs) {
  for (auto &rhs_blk : rhs.blocks_) {
    auto has_blk =  false;
    for (auto &lhs_blk : blocks_) {
      if (lhs_blk->QNSectorSetHash() == rhs_blk->QNSectorSetHash()) {
        assert(lhs_blk->size == rhs_blk->size);
        auto lhs_blk_data = lhs_blk->DataRef();
        auto rhs_blk_data = rhs_blk->DataConstRef();
        for (long i = 0; i < lhs_blk->size; i++) {
           lhs_blk_data[i] -= rhs_blk_data[i];
        }
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto pnew_blk = new QNBlock(rhs_blk->qnscts);
      auto pnew_data = pnew_blk->DataRef();
      auto rhs_blk_data = rhs_blk->DataConstRef();
      for (long i = 0; i < rhs_blk->size; i++) {
        pnew_data[i] = -rhs_blk_data[i];
      }
      blocks_.push_back(pnew_blk);
    }
  }
  return this;
}


bool GQTensor::operator==(const GQTensor &rhs) const {
  // Indexes check.
  if (indexes != rhs.indexes) {
    return false;
  }
  // Scalar check.
  if (indexes.size() == 0 && rhs.indexes.size() == 0 && scalar != rhs.scalar) {
    return false;
  }
  // Block number check.
  if (blocks_.size() != rhs.blocks_.size()) {
    return false;
  }
  // Blocks check.
  for (auto &lhs_blk : blocks_) {
    auto has_eq_blk = false;
    for (auto &rhs_blk : rhs.blocks_) {
      if (rhs_blk->QNSectorSetHash() == lhs_blk->QNSectorSetHash()) {
        if (!ArrayEq(
                 rhs_blk->DataConstRef(), rhs_blk->size,
                 lhs_blk->DataConstRef(), lhs_blk->size)) {
          return false;
        } else {
          has_eq_blk = true;
          break;
        }
      }
    }
    if (!has_eq_blk) {
      return false;
    }
  }
  return true;
}


// Iterators.
// Generate all coordinates. Cost so much. Be careful to use.
std::vector<std::vector<long>> GQTensor::CoorsIter(void) const {
  return GenAllCoors(shape);
}


// Private members.
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


double GQTensor::Norm(void) {
  double norm2 = 0.0; 
  for (auto &b : blocks_) {
    for (long i = 0; i < b->size; ++i) {
      norm2 += std::pow(b->DataRef()[i], 2.0);
    }
  }
  return std::sqrt(norm2);
}


// GQTensor objects operations.
Index InverseIndex(const Index &idx) {
  Index inversed_idx = idx;
  inversed_idx.Dag();
  return inversed_idx;
}


GQTensor Dag(const GQTensor &t) {
  GQTensor dag_t(t);
  dag_t.Dag();
  /* TODO: use move to improve the performance. */
  return dag_t;
}


QN Div(const GQTensor &t) {
  auto blks = t.BlksConstRef();
  auto blk_num = blks.size();
  QN div = CalcDiv(blks[0]->qnscts, t.indexes);
  for (size_t i = 1; i < blk_num; ++i) {
    auto blki_div = CalcDiv(blks[i]->qnscts, t.indexes);
    if (blki_div != div) {
      std::cout << "Tensor does not have a special divergence. Return QN()." << std::endl;
      return QN();
    }
  }
  return div;
}


GQTensor operator*(const GQTensor &t, const double &s) {
  auto muled_t = GQTensor(t);
  // For scalar case.
  if (muled_t.indexes.size() == 0) {
    muled_t.scalar *= s;
    return muled_t;
  }
  // For tensor case.
  for (auto &blk : muled_t.BlksRef()) {
    auto data = blk->DataRef();
    for (long i = 0; i < blk->size; ++i) {
      data[i]  = data[i] * s;
    }
  }
  return muled_t;
}


GQTensor operator*(const double &s, const GQTensor &t) { return t * s; }


std::ifstream &bfread(std::ifstream &ifs, GQTensor &t) {
  long ndim;
  ifs >> ndim;
  t.indexes = std::vector<Index>(ndim);
  for (auto &idx : t.indexes) { bfread(ifs, idx); }
  t.shape = std::vector<long>(ndim);
  for (auto &order : t.shape) { ifs >> order; }
  ifs >> t.scalar;

  long blk_num;
  ifs >> blk_num;
  for (auto &blk : t.blocks_) { delete blk; }
  t.blocks_ = std::vector<QNBlock *>(blk_num);
  for (auto &blk : t.blocks_) {
    blk = new QNBlock();
    bfread(ifs, *blk);
  }
  return ifs;
}


std::ofstream &bfwrite(std::ofstream &ofs, const GQTensor &t) {
  long ndim = t.indexes.size();
  ofs << ndim << std::endl;
  for (auto &idx : t.indexes) { bfwrite(ofs, idx); }
  for (auto &order : t.shape) { ofs << order << std::endl; }
  ofs << t.scalar << std::endl;

  long blk_num = t.blocks_.size();
  ofs << blk_num << std::endl;
  for (auto &blk : t.blocks_) { bfwrite(ofs, *blk); }
  return ofs;
}


// Helper functions.
QN CalcDiv(const std::vector<QNSector> &qnscts, const std::vector<Index> &indexes) {
  QN div;
  auto ndim = indexes.size();
  assert(qnscts.size() == ndim);
  for (size_t i = 0; i < ndim; ++i) {
    if (indexes[i].dir == IN) {
      auto qnflow = -qnscts[i].qn;
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
      auto qnflow = qnscts[i].qn;
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


QN CalcDiv(const QNSectorSet &blk_key, const std::vector<Index> &indexes) {
  return CalcDiv(blk_key.qnscts, indexes);
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
