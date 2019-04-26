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
  std::size_t hash(void) const;
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


class QNSector {
public:
  QNSector(void) { qn = QN(), dim = 0; }
  QNSector(const QN &qn, const long &dim) : qn(qn), dim(dim) {}
  size_t hash(void) const;
  QN qn;
  long dim;
private:
  std::hash<int> int_hasher_;
};

bool operator==(const QNSector &, const QNSector &);

bool operator!=(const QNSector &, const QNSector &);

} /* gqten */ 
#endif /* ifndef GQTEN_GQTEN_H */
