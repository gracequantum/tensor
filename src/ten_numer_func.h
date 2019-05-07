/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-05 14:11
* 
* Description: GraceQ/tensor project. Numerical functions for GQTensor, head file.
*/
#ifndef GQTEN_TEN_NUMER_FUNC_H
#define GQTEN_TEN_NUMER_FUNC_H


#include "gqten/gqten.h"

#include <vector>


namespace gqten {


GQTensor InitCtrctedTen(
    const GQTensor &, const GQTensor &,
    const std::vector<long> &, const std::vector<long> &);

QNBlock *ContractBlock(
    const QNBlock *, const QNBlock *,
    const std::vector<long> &, const std::vector<long> &);

struct BlkCtrctInfo {
public:
  BlkCtrctInfo(
      const double *data,
      const long &savedim, const long &ctrctdim,
      const std::vector<QNSector> &saved_qnscts) :
    data(data),
    savedim(savedim), ctrctdim(ctrctdim),
    saved_qnscts(saved_qnscts) {}
  const double *data = nullptr;
  const long savedim = 1;
  const long ctrctdim = 1;
  const std::vector<QNSector> saved_qnscts;
};

BlkCtrctInfo BlkCtrctPreparer(
    const QNBlock &, const std::vector<long> &, const std::string &);

double *TransposeData(
    const double *,
    const long &,
    const long &,
    const std::vector<long> &,
    const std::vector<long> &);

double *MatMul(
    const double *, const long &, const long &,
    const double *, const long &, const long &);
} /* gqten */ 
#endif /* ifndef GQTEN_TEN_NUMER_FUNC_H */
