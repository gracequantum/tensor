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


namespace gqten {


// GQTensor storage file suffix.
const std::string kGQTenFileSuffix = "gqten";

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
  QN(const QN &);
  QN &operator=(const QN &);

  std::size_t Hash(void) const;
  QN operator-(void) const;
  QN &operator+=(const QN &);

private:
  std::vector<std::string> names_; 
  std::vector<long> values_;
  std::size_t hash_;

  std::size_t CalcHash(void) const;
};

bool operator==(const QN &, const QN &);

bool operator!=(const QN &, const QN &);

QN operator+(const QN &, const QN &);

QN operator-(const QN &, const QN &);

std::ifstream &bfread(std::ifstream &, QN &);

std::ofstream &bfwrite(std::ofstream &, const QN &);


// Quantum number sector.
class QNSector {
friend std::ifstream &bfread(std::ifstream &, QNSector &);
friend std::ofstream &bfwrite(std::ofstream &, const QNSector &);

public:
  QNSector(const QN &qn, const long &dim) : qn(qn), dim(dim) {
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
  QNSectorSet(const std::vector<QNSector> & qnscts) : qnscts(qnscts) {}
  QNSectorSet(const QNSectorSet &qnss) : qnscts(qnss.qnscts) {}
  QNSectorSet(const std::vector<const QNSector*> &);

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
  Index(
      const std::vector<QNSector> &qnscts) :
          Index(qnscts, NDIR, "") {}
  Index(
      const std::vector<QNSector> &qnscts,
      const std::string &dir) : Index(qnscts, dir, "") {}

  size_t Hash(void) const override;
  InterOffsetQnsct CoorOffsetAndQnsct(long) const;

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
class QNBlock : public QNSectorSet {
friend std::ifstream &bfread(std::ifstream &, QNBlock &);
friend std::ofstream &bfwrite(std::ofstream &, const QNBlock &);

public:
  QNBlock(void) = default;
  QNBlock(const std::vector<QNSector> &);
  QNBlock(const std::vector<const QNSector *> &);

  QNBlock(const QNBlock &);
  QNBlock &operator=(const QNBlock &);

  ~QNBlock(void) override;
  
  // Element getter and setter.
  const double &operator()(const std::vector<long> &) const;
  double &operator()(const std::vector<long> &);

  size_t PartHash(const std::vector<long> &) const;
  size_t QNSectorSetHash(void) const { return qnscts_hash_; }

  // Data access.
  const double *DataConstRef(void) const { return data_; }
  double * &DataRef(void) { return data_; }

  // Inplace operations.
  void Random(void);
  void Transpose(const std::vector<long> &);

  long ndim = 0;
  std::vector<long> shape;
  long size = 0;

private:
  double *data_ = nullptr;    // Data in a 1D array.
  std::vector<long> data_offsets_;
  std::size_t qnscts_hash_ = 0;
};

std::ifstream &bfread(std::ifstream &, QNBlock &);

std::ofstream &bfwrite(std::ofstream &, const QNBlock &);


// Tensor with U1 symmetry.
struct BlkCoorsAndBlkKey {
  BlkCoorsAndBlkKey(
      const std::vector<long> &blk_coors, const QNSectorSet &blk_key) :
      blk_coors(blk_coors), blk_key(blk_key) {}
  std::vector<long> blk_coors;
  QNSectorSet blk_key;
};

class GQTensor {
friend std::ifstream &bfread(std::ifstream &, GQTensor &);
friend std::ofstream &bfwrite(std::ofstream &, const GQTensor &);

public:
  GQTensor(void) = default;
  GQTensor(const std::vector<Index> &);

  GQTensor(const GQTensor &);
  GQTensor &operator=(const GQTensor &);

  ~GQTensor(void);

  // Element getter and setter.
  double Elem(const std::vector<long> &) const;
  double &operator()(const std::vector<long> &);

  // Inplace operations.
  void Random(const QN &);
  void Transpose(const std::vector<long> &);
  double Normalize(void);
  void Dag(void) { for (auto &index : indexes) { index.Dag(); } }

  // Operators overload.
  GQTensor operator-(void) const;

  GQTensor operator+(const GQTensor &);
  GQTensor &operator+=(const GQTensor &);

  bool operator==(const GQTensor &) const;
  bool operator!=(const GQTensor &rhs) const { return !(*this == rhs); }

  // Access to the blocks.
  const std::vector<QNBlock *> &BlksConstRef(void) const { return blocks_; }
  std::vector<QNBlock *> &BlksRef(void) { return blocks_; }

  // Iterators.
  // Return all the tensor coordinates. So heavy that you should not use it!
  std::vector<std::vector<long>> CoorsIter(void) const;

  // Public data members.
  std::vector<Index> indexes;
  double scalar = 0.0;
  std::vector<long> shape;

private:
  std::vector<QNBlock *> blocks_;

  double Norm(void);

  BlkCoorsAndBlkKey TargetBlkCoorsAndBlkKey(const std::vector<long> &) const;
  std::vector<QNSectorSet> BlkKeysIter(void) const;
};


// GQTensor objects operations.
// For Index.
Index InverseIndex(const Index &);

// For GQTensor.
GQTensor Dag(const GQTensor &);

// Just mock the dag. Not construct a new object.
inline const GQTensor &MockDag(const GQTensor &t) { return t; }

QN Div(const GQTensor &);

GQTensor operator*(const GQTensor &, const double &);

GQTensor operator*(const double &, const GQTensor &);

// GQTensor I/O
std::ifstream &bfread(std::ifstream &, GQTensor &);

std::ofstream &bfwrite(std::ofstream &, const GQTensor &);


// Tensor numerical functions.
// Tensors contraction.
GQTensor *Contract(
    const GQTensor &, const GQTensor &,
    const std::vector<std::vector<long>> &);

// Tensors linear combination.
/* TODO: For scalar tensor case. */
void LinearCombine(
    const std::vector<double> &,
    const std::vector<GQTensor *> &,
    GQTensor *);

void LinearCombine(
    const std::size_t,
    const double *,
    const std::vector<GQTensor *> &,
    GQTensor *);

// Tensor SVD.
struct SvdRes {
  SvdRes(
      GQTensor *u, GQTensor *s, GQTensor *v,
      const double trunc_err, const long D) :
      u(u), s(s), v(v), trunc_err(trunc_err), D(D) {}
  GQTensor *u;
  GQTensor *s;
  GQTensor *v;
  const double trunc_err;
  const long D;
};

SvdRes Svd(
    const GQTensor &,
    const long &, const long &,
    const QN &, const QN &,
    const double &, const long &, const long &);


// Helper functions.
QN CalcDiv(const QNSectorSet &, const std::vector<Index> &);

QN CalcDiv(const std::vector<QNSector> &, const std::vector<Index> &);

std::vector<long> CalcDataOffsets(const std::vector<long> &);

long MulToEnd(const std::vector<long> &, int);

double *TransposeData(
    const double *,
    const long &,
    const long &,
    const std::vector<long> &,
    const std::vector<long> &);

std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &);


// Some function templates.
template<typename T>
T CalcCartProd(T v) {
  T s = {{}};
  for (const auto &u : v) {
    T r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}


// Inline functions.
inline long CalcOffset(
    const std::vector<long> &coors,
    const long ndim,
    const std::vector<long> &data_offsets) {
  long offset = 0;
  for (long i = 0; i < ndim; ++i) {
    offset += coors[i] * data_offsets[i];
  }
  return offset;
}


inline bool ArrayEq(
    const double *aptr1, const size_t size1,
    const double *aptr2, const size_t size2) {
  if (size1 !=  size2) {
    return false;
  }
  for (size_t i = 0; i < size1; ++i) {
    if (aptr1[i] != aptr2[i]) {
      return false;
    }
  }
  return true;
}


// Timer.
class Timer {
public:
  Timer(const std::string &);

  void Restart(void);
  double Elapsed(void);
  void PrintElapsed(void);

private:
  double start_;
  std::string notes_;

  double GetWallTime(void);
};
} /* gqten */ 
#endif /* ifndef GQTEN_GQTEN_H */
