// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:33
* 
* Description: GraceQ/tensor project. The main header file.
*/
#ifndef GQTEN_GQTEN_H
#define GQTEN_GQTEN_H


#include <string>
#include <vector>
#include <fstream>
#include <cmath>

#include "gqten/detail/fwd_dcl.h"
#include "gqten/detail/consts.h"
#include "gqten/detail/value_t.h"


namespace gqten {


// Quantum number.
struct QNNameVal {
  QNNameVal() = default;
  QNNameVal(const std::string &nm, const long val): name(nm), val(val) {}

  std::string name;
  long val;
};

class QN {
friend std::ifstream &bfread(std::ifstream &, QN &);
friend std::ofstream &bfwrite(std::ofstream &, const QN &);

public:
  QN(void);
  QN(const std::vector<QNNameVal> &);
  QN(const std::vector<long> &);

  QN(const QN &);
  QN &operator=(const QN &);

  std::size_t Hash(void) const;

  QN operator-(void) const;
  QN &operator+=(const QN &);

private:
  std::vector<long> values_;
  std::size_t hash_;

  std::size_t CalcHash(void) const;
};

QN operator+(const QN &, const QN &);

QN operator-(const QN &, const QN &);

bool operator==(const QN &, const QN &);

bool operator!=(const QN &, const QN &);

std::ifstream &bfread(std::ifstream &, QN &);

std::ofstream &bfwrite(std::ofstream &, const QN &);


// Quantum number sector.
class QNSector {
friend std::ifstream &bfread(std::ifstream &, QNSector &);
friend std::ofstream &bfwrite(std::ofstream &, const QNSector &);

public:
  QNSector(const QN &qn, const long dim) : qn(qn), dim(dim) {
    hash_ = CalcHash();
  }
  QNSector(void) : QNSector(QN(), 0) {}

  QNSector(const QNSector &qns) : qn(qns.qn), dim(qns.dim), hash_(qns.hash_) {}
  QNSector &operator=(const QNSector &rhs);

  size_t Hash(void) const { return hash_; }

  QN qn;
  long dim;

private:
  size_t CalcHash(void) const { return qn.Hash() ^ dim; }
  size_t hash_;
};

bool operator==(const QNSector &, const QNSector &);

bool operator!=(const QNSector &, const QNSector &);

std::ifstream &bfread(std::ifstream &, QNSector &);

std::ofstream &bfwrite(std::ofstream &, const QNSector &);


// Quantum number sector set.
class QNSectorSet {
public:
  QNSectorSet(void) {}
  QNSectorSet(const std::vector<QNSector> &qnscts) : qnscts(qnscts) {}
  QNSectorSet(const std::vector<const QNSector*> &);

  QNSectorSet(const QNSectorSet &qnss) : qnscts(qnss.qnscts) {}

  virtual ~QNSectorSet() = default;

  virtual size_t Hash(void) const;

  std::vector<QNSector> qnscts;
};

bool operator==(const QNSectorSet &, const QNSectorSet &);

bool operator!=(const QNSectorSet &, const QNSectorSet &);


// Index.
#define NDIR "NDIR"
#define IN "IN"
#define OUT "OUT"

struct InterOffsetQnsct {
  InterOffsetQnsct(const long &inter_offset, const QNSector &qnsct) :
      inter_offset(inter_offset), qnsct(qnsct) {}
  long inter_offset;
  QNSector qnsct;
};

class Index : public QNSectorSet {
friend std::ifstream &bfread(std::ifstream &, Index &);
friend std::ofstream &bfwrite(std::ofstream &, const Index &);

public:
  Index(void) : QNSectorSet(), dim(0), dir(NDIR), tag("") {}

  Index(
      const std::vector<QNSector> &qnscts,
      const std::string &dir,
      const std::string &tag) : QNSectorSet(qnscts), dir(dir), tag(tag) {
    dim = CalcDim(); 
  }
  Index(const std::vector<QNSector> &qnscts) : Index(qnscts, NDIR, "") {}
  Index(const std::vector<QNSector> &qnscts, const std::string &dir) :
      Index(qnscts, dir, "") {}

  Index(const Index &index) :
      QNSectorSet(index.qnscts),
      dim(index.dim), dir(index.dir), tag(index.tag) {}
  Index &operator=(const Index &rhs) {
    qnscts = rhs.qnscts;
    dim = rhs.dim;
    dir = rhs.dir;
    tag = rhs.tag;
    return *this;
  }

  size_t Hash(void) const override;
  InterOffsetQnsct CoorInterOffsetAndQnsct(const long) const;

  // Inplace operations.
  void Dag(void) {
    if (dir == IN) {
      dir = OUT;
    } else if (dir == OUT) {
      dir = IN;
    }
  }

  // Operators overloading.
  bool operator==(const Index &rhs) const { return  Hash() ==  rhs.Hash(); }

  long CalcDim(void) {
    long dim = 0;
    for (auto &qnsct : qnscts) {
      dim += qnsct.dim;
    }
    return dim;
  }

  long dim;
  std::string dir;
  std::string tag;
};

std::ifstream &bfread(std::ifstream &, Index &);

std::ofstream &bfwrite(std::ofstream &, const Index &);


// Dense block labeled by the quantum number.
template <typename ElemType>
class QNBlock : public QNSectorSet {
// Binary I/O.
friend std::ifstream &bfread<ElemType>(std::ifstream &, QNBlock<ElemType> &);
friend std::ofstream &bfwrite<ElemType>(std::ofstream &, const QNBlock<ElemType> &);
// Some functions called by tensor numerical functions to use the private constructor.
friend std::vector<QNBlock<ElemType> *> BlocksCtrctBatch<ElemType>(
    const std::vector<long> &, const std::vector<long> &,
    const ElemType,
    const std::vector<QNBlock<ElemType> *> &,
    const std::vector<QNBlock<ElemType> *> &);

public:
  QNBlock(void) = default;
  QNBlock(const std::vector<QNSector> &);

  QNBlock(const QNBlock &);
  QNBlock &operator=(const QNBlock &);
  
  QNBlock(QNBlock &&) noexcept;
  QNBlock &operator=(QNBlock &&) noexcept;

  ~QNBlock(void) override;
  
  // Element getter and setter.
  const ElemType &operator()(const std::vector<long> &) const;
  ElemType &operator()(const std::vector<long> &);

  // Data access.
  const ElemType *cdata(void) const { return data_; }   // constant reference.
  ElemType * &data(void) { return data_; }              // non-constant reference.

  // Hash methods.
  size_t PartHash(const std::vector<long> &) const;
  size_t QNSectorSetHash(void) const { return qnscts_hash_; }

  // Inplace operations.
  void Random(void);
  void Transpose(const std::vector<long> &);

  // Public data members.
  long ndim = 0;              // Number of dimensions.
  std::vector<long> shape;    // Shape of the block.
  long size = 0;              // Total number of elements in this block.

private:
  // NOTE: For performance reason, this constructor will NOT initialize the data_ to 0!!!
  // It should only be intra-used.
  QNBlock(const std::vector<const QNSector *> &);

  ElemType *data_ = nullptr;    // Data in a 1D array.
  std::vector<long> data_offsets_;
  std::size_t qnscts_hash_ = 0;
};


// Tensor with U1 symmetry.
struct BlkInterOffsetsAndQNSS {     // QNSS: QNSectorSet.
  BlkInterOffsetsAndQNSS(
      const std::vector<long> &blk_inter_offsets, const QNSectorSet &blk_qnss) :
      blk_inter_offsets(blk_inter_offsets), blk_qnss(blk_qnss) {}

  std::vector<long> blk_inter_offsets;
  QNSectorSet blk_qnss;
};


template <typename ElemType>
class GQTensor {
friend std::ifstream &bfread<ElemType>(std::ifstream &, GQTensor<ElemType> &);
friend std::ofstream &bfwrite<ElemType>(std::ofstream &, const GQTensor<ElemType> &);

public:
  GQTensor(void) = default;
  GQTensor(const std::vector<Index> &);

  GQTensor(const GQTensor &);
  GQTensor &operator=(const GQTensor &);

  GQTensor(GQTensor &&) noexcept;
  GQTensor &operator=(GQTensor &&) noexcept;

  ~GQTensor(void);

  // Element getter and setter.
  ElemType Elem(const std::vector<long> &) const;     // Getter.
  ElemType &operator()(const std::vector<long> &);    // Setter.

  // Access to the blocks.
  const std::vector<QNBlock<ElemType> *> &cblocks(void) const {
    return blocks_;
  }
  std::vector<QNBlock<ElemType> *> &blocks(void) { return blocks_; }

  // Inplace operations.

  // Random set tensor elements with given quantum number divergence.
  // Any original blocks will be destroyed.
  void Random(const QN &);

  // Tensor transpose.
  void Transpose(const std::vector<long> &);

  // Normalize the GQTensor and return its norm.
  GQTEN_Double Normalize(void);

  // Switch the direction of the indexes, complex conjugate of the element.
  void Dag(void);

  // Operators overload.
  GQTensor operator-(void) const;
  GQTensor operator+(const GQTensor &);
  GQTensor &operator+=(const GQTensor &);

  bool operator==(const GQTensor &) const;
  bool operator!=(const GQTensor &rhs) const { return !(*this == rhs); }

  // Iterators.
  // Return all the tensor coordinates. So heavy that you should not use it!
  std::vector<std::vector<long>> CoorsIter(void) const;

  // Public data members.
  std::vector<Index> indexes;
  ElemType scalar = 0.0;
  std::vector<long> shape;

private:
  std::vector<QNBlock<ElemType> *> blocks_;

  double Norm(void);

  BlkInterOffsetsAndQNSS CalcTargetBlkInterOffsetsAndQNSS(
      const std::vector<long> &) const;
  std::vector<QNSectorSet> BlkQNSSsIter(void) const;
};


// GQTensor objects operations.
// For Index.
Index InverseIndex(const Index &);

// For GQTensor.
template <typename ElemType>
GQTensor<ElemType> Dag(const GQTensor<ElemType> &);

// Just mock the dag. Not construct a new object.
template <typename ElemType>
inline const GQTensor<ElemType> &MockDag(const GQTensor<ElemType> &t) {
  return t;
}

template <typename ElemType>
QN Div(const GQTensor<ElemType> &);

template <typename ElemType>
GQTensor<ElemType> operator*(const GQTensor<ElemType> &, const ElemType &);

template <typename ElemType>
GQTensor<ElemType> operator*(const ElemType &, const GQTensor<ElemType> &);

GQTensor<GQTEN_Complex> ToComplex(const GQTensor<GQTEN_Double> &);


// Tensor numerical functions.
// Tensors contraction.
template <typename TenElemType>
void Contract(
    const GQTensor<TenElemType> *, const GQTensor<TenElemType> *,
    const std::vector<std::vector<long>> &,
    GQTensor<TenElemType> *);

// This API just for forward compatibility, it will be deleted soon.
// TODO: Remove these API.
inline DGQTensor *Contract(
    const DGQTensor &ta, const DGQTensor &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new DGQTensor();
  Contract(&ta, &tb, axes_set, res_t);
  return res_t;
}

inline ZGQTensor *Contract(
    const ZGQTensor &ta, const ZGQTensor &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new ZGQTensor();
  Contract(&ta, &tb, axes_set, res_t);
  return res_t;
}

inline ZGQTensor *Contract(
    const DGQTensor &ta, const ZGQTensor &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new ZGQTensor();
  auto zta = ToComplex(ta);
  Contract(&zta, &tb, axes_set, res_t);
  return res_t;
}

inline ZGQTensor *Contract(
    const ZGQTensor &ta, const DGQTensor &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new ZGQTensor();
  auto ztb = ToComplex(tb);
  Contract(&ta, &ztb, axes_set, res_t);
  return res_t;
}


// Tensors linear combination.
// Do the operation: res += (coefs[0]*ts[0] + coefs[1]*ts[1] + ...).
//[> TODO: Support scalar (rank 0) tensor case. <]
template <typename TenElemType>
void LinearCombine(
    const std::vector<TenElemType> &,
    const std::vector<GQTensor<TenElemType> *> &,
    GQTensor<TenElemType> *);

template <typename TenElemType>
void LinearCombine(
    const std::size_t,
    const TenElemType *,
    const std::vector<GQTensor<TenElemType> *> &,
    GQTensor<TenElemType> *);

inline void LinearCombine(
    const std::size_t size,
    const double *dcoefs,
    const std::vector<GQTensor<GQTEN_Complex> *> &zts,
    GQTensor<GQTEN_Complex> *res) {
  auto zcoefs = new GQTEN_Complex [size];
  for (size_t i = 0; i < size; ++i) {
    zcoefs[i] = dcoefs[i];
  }
  LinearCombine(size, zcoefs, zts, res);
  delete [] zcoefs;
}


// Tensor SVD.
template <typename TenElemType>
void Svd(
    const GQTensor<TenElemType> *,
    const long, const long,
    const QN &, const QN &,
    const double, const long, const long,
    GQTensor<TenElemType> *,
    GQTensor<GQTEN_Double> *,
    GQTensor<TenElemType> *,
    double *, long *);


// These APIs just for forward compatibility, it will be deleted soon.
// TODO: Remove these APIs.
template <typename TenElemType>
struct SvdRes {
  SvdRes(
      GQTensor<TenElemType> *u,
      GQTensor<GQTEN_Double> *s,
      GQTensor<TenElemType> *v,
      const double trunc_err, const long D) :
      u(u), s(s), v(v), trunc_err(trunc_err), D(D) {}
  GQTensor<TenElemType> *u;
  GQTensor<GQTEN_Double> *s;
  GQTensor<TenElemType> *v;
  const double trunc_err;
  const long D;
};


template <typename TenElemType>
inline SvdRes<TenElemType> Svd(
    const GQTensor<TenElemType> &t,
    const long ldims, const long rdims,
    const QN &ldiv, const QN &rdiv,
    const double cutoff, const long Dmin, const long Dmax) {
  auto pu =  new GQTensor<TenElemType>();
  auto ps =  new GQTensor<GQTEN_Double>();
  auto pvt = new GQTensor<TenElemType>();
  double trunc_err;
  long D;
  Svd(
      &t,
      ldims, rdims,
      ldiv, rdiv,
      cutoff, Dmin, Dmax,
      pu, ps, pvt,
      &trunc_err, &D);
  return SvdRes<TenElemType>(pu, ps, pvt,trunc_err, D);
}


template <typename TenElemType>
inline SvdRes<TenElemType> Svd(
    const GQTensor<TenElemType> &t,
    const long ldims, const long rdims,
    const QN &ldiv, const QN &rdiv) {
  auto t_shape = t.shape;
  long lsize = 1;
  long rsize = 1;
  for (std::size_t i = 0; i < t_shape.size(); ++i) {
    if (i < ldims) {
      lsize *= t_shape[i];
    } else {
      rsize *= t_shape[i];
    }
  }
  auto D = ((lsize >= rsize) ? lsize : rsize);
  return Svd(
      t,
      ldims, rdims,
      ldiv, rdiv,
      0, D, D);
}


// Tensor transpose function multi-thread controller.
int GQTenGetTensorTransposeNumThreads(void);

void GQTenSetTensorTransposeNumThreads(const int);


// Timer.
class Timer {
public:
  Timer(const std::string &);

  void Restart(void);
  double Elapsed(void);
  double PrintElapsed(std::size_t precision = 5);

private:
  double start_;
  std::string notes_;

  double GetWallTime(void);
};
} /* gqten */ 


// Include implementation details.
#include "gqten/detail/qnblock_impl.h"
#include "gqten/detail/gqtensor_impl.h"
#include "gqten/detail/ten_ctrct_impl.h"
#include "gqten/detail/ten_lincmb_impl.h"
#include "gqten/detail/ten_svd_impl.h"


#endif /* ifndef GQTEN_GQTEN_H */
