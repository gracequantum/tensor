/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:33
* 
* Description: GraceQ/tensor project. The main header file.
*/
#ifndef GQTEN_GQTEN_H
#define GQTEN_GQTEN_H


#include <string>
#include <initializer_list>
#include <vector>


namespace gqten {


// Quantum number.
struct QNNameVal {
  QNNameVal() = default;
  QNNameVal(const std::string &nm, const int &val): name(nm), val(val) {}
  std::string name;
  int val;
};

using QNNameValIniter = std::initializer_list<QNNameVal>;

class QN {
public:
  QN() = default;
  QN(QNNameValIniter); 
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
  QNSectorSet(const std::initializer_list<QNSector> & init_qnscts) : qnscts(init_qnscts) {}
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
      const std::initializer_list<QNSector> &init_qnscts,
      const std::string &dir,
      const std::string &tag) : QNSectorSet(init_qnscts), dir(dir), tag(tag) {}
  Index(
      const std::initializer_list<QNSector> &init_qnscts) :
          Index(init_qnscts, NDIR, "") {}
  Index(
      const std::initializer_list<QNSector> &init_qnscts,
      const std::string &dir) : Index(init_qnscts, dir, "") {}
  size_t Hash(void) const override;
  InterOffsetQnsct CoorOffsetAndQnsct(long) const;
  std::string dir = NDIR;
  std::string tag = "";

private:
  std::hash<std::string> str_hasher_;
};


// Dense block labeled by the quantum number.
class QNBlock : public QNSectorSet {
public:
  QNBlock() = default;
  QNBlock(const std::initializer_list<QNSector> &);
  QNBlock(const std::vector<QNSector> &);
  ~QNBlock(void);
  
  const double &operator()(const std::initializer_list<long> &) const;
  const double &operator()(const std::vector<long> &) const;
  double &operator()(const std::initializer_list<long> &);
  double &operator()(const std::vector<long> &);
  size_t PartHash(const std::initializer_list<long> &) const;
  void Random(void);
  const double *DataConstRef(void) const { return data_; }
  double *DataRef(void) const { return data_; }

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
  GQTensor() = default;
  GQTensor(const std::initializer_list<Index> &idxes) : indexes(idxes) {}
  ~GQTensor(void);

  double Elem(const std::initializer_list<long> &) const;
  double &operator()(const std::initializer_list<long> &);
  void Random(const QN &);
  const std::vector<QNBlock *> &BlksConstRef(void) const { return blocks_; }
  std::vector<QNBlock *> BlksRef(void) const { return blocks_; }

  std::vector<Index> indexes;
  double scalar = 0.0;

private:
  std::vector<QNBlock *> blocks_;
  BlkCoorsAndBlkKey TargetBlkCoorsAndBlkKey(const std::vector<long> &) const;
  std::vector<QNSectorSet> BlkKeysIter(void) const;
};


// GQTensor objects operations.
Index InverseIndex(const Index &);


// Helper functions.
QN CalcDiv(const QNSectorSet &, const std::vector<Index> &);

std::vector<long> CalcDataOffsets(const std::vector<long> &);

long MulToEnd(const std::vector<long> &, int);
} /* gqten */ 
#endif /* ifndef GQTEN_GQTEN_H */
