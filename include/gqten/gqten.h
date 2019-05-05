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


namespace gqten {


// Quantum number.
struct QNNameVal {
  QNNameVal() = default;
  QNNameVal(const std::string &nm, const int &val): name(nm), val(val) {}
  std::string name;
  int val;
};

class QN {
public:
  QN() = default;
  QN(const std::vector<QNNameVal> &);
  QN(const QN &);
  std::size_t Hash(void) const;
  QN operator-(void) const;
  QN &operator+=(const QN &);

private:
  std::vector<std::string> names_; 
  std::vector<int> values_;
};

bool operator==(const QN &, const QN &);

bool operator!=(const QN &, const QN &);

QN operator+(const QN &, const QN &);

QN operator-(const QN &, const QN &);


// Quantum number sector.
class QNSector {
public:
  QNSector() = default;
  QNSector(const QN &qn, const long &dim) : qn(qn), dim(dim) {}
  size_t Hash(void) const;
  QN qn = QN();
  long dim = 0;
private:
  std::hash<int> int_hasher_;
};

bool operator==(const QNSector &, const QNSector &);

bool operator!=(const QNSector &, const QNSector &);


// Quantum number sector set.
class QNSectorSet {
public:
  QNSectorSet() = default;
  QNSectorSet(const std::vector<QNSector> & qnscts) : qnscts(qnscts) {}
  std::vector<QNSector> qnscts;
  virtual size_t Hash(void) const;
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
public:
  Index() = default;
  Index(
      const std::vector<QNSector> &qnscts,
      const std::string &dir,
      const std::string &tag) : QNSectorSet(qnscts), dir(dir), tag(tag) {}
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
  bool operator==(const Index &rhs) { return  Hash() ==  rhs.Hash(); }

  std::string dir = NDIR;
  std::string tag = "";

private:
  std::hash<std::string> str_hasher_;
};


// Dense block labeled by the quantum number.
class QNBlock : public QNSectorSet {
public:
  QNBlock(void) {}
  QNBlock(const std::vector<QNSector> &);

  QNBlock(const QNBlock &);
  QNBlock &operator=(const QNBlock &);

  ~QNBlock(void);
  
  const double &operator()(const std::vector<long> &) const;
  double &operator()(const std::vector<long> &);

  size_t PartHash(const std::vector<long> &) const;

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
};


// Tensor with U1 symmetry.
struct BlkCoorsAndBlkKey {
  BlkCoorsAndBlkKey(
      const std::vector<long> &blk_coors, const QNSectorSet &blk_key) :
      blk_coors(blk_coors), blk_key(blk_key) {}
  std::vector<long> blk_coors;
  QNSectorSet blk_key;
};

class GQTensor {
public:
  GQTensor(void) {}
  GQTensor(const std::vector<Index> &idxes) : indexes(idxes) {
    for (auto &index : indexes) {
      auto size = 0;
      for (auto &qnsct : index.qnscts) {
        size += qnsct.dim;
      }
      shape.push_back(size);
    }
  }

  GQTensor(const GQTensor &);
  GQTensor &operator=(const GQTensor &);

  ~GQTensor(void);

  // Element getter and setter.
  double Elem(const std::vector<long> &) const;
  double &operator()(const std::vector<long> &);

  // Inplace operations.
  void Random(const QN &);
  void Transpose(const std::vector<long> &);
  void Normalize(void);
  void Dag(void) { for (auto &index : indexes) { index.Dag(); } }

  // Operators overload.
  GQTensor operator-(void) const;
  GQTensor operator+(const GQTensor &);
  GQTensor operator-(const GQTensor &rhs) { return *this + (-rhs);}
  bool operator==(const GQTensor &);
  bool operator!=(const GQTensor &rhs) { return !(*this == rhs); }

  // Access to the blocks.
  const std::vector<QNBlock *> &BlksConstRef(void) const { return blocks_; }
  std::vector<QNBlock *> &BlksRef(void) { return blocks_; }

  // Iterators.
  std::vector<std::vector<long>> CoorsIter(void) const;

  // Public data members.
  std::vector<Index> indexes;
  double scalar = 0.0;
  std::vector<long> shape;

private:
  std::vector<QNBlock *> blocks_;

  BlkCoorsAndBlkKey TargetBlkCoorsAndBlkKey(const std::vector<long> &) const;
  std::vector<QNSectorSet> BlkKeysIter(void) const;
  double Norm(void);
};


// GQTensor objects operations.
Index InverseIndex(const Index &);

GQTensor Dag(const GQTensor &);

GQTensor operator*(const GQTensor &, const double &);

GQTensor operator*(const double &, const GQTensor &);


// Helper functions.
QN CalcDiv(const QNSectorSet &, const std::vector<Index> &);

std::vector<long> CalcDataOffsets(const std::vector<long> &);

long MulToEnd(const std::vector<long> &, int);

void TransposeBlkData(
    double * &, const long &, const long &,
    const std::vector<long> &,
    const std::vector<long> &, const std::vector<long> &,
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


inline std::vector<long> TransCoors(
    const std::vector<long> &old_coors, const std::vector<long> &axes_map) {
  std::vector<long> new_coors(old_coors.size());
  for (size_t i = 0; i < axes_map.size(); ++i) {
    new_coors[i] = old_coors[axes_map[i]];
  }
  return new_coors;
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
} /* gqten */ 
#endif /* ifndef GQTEN_GQTEN_H */
