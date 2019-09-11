// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 17:07
* 
* Description: GraceQ/tensor project. Forward declaration for intra-used utility functions.
*/
#ifndef GQTEN_UTILS_H
#define GQTEN_UTILS_H


#include "gqten/gqten.h"

#include <vector>


namespace gqten {


QN CalcDiv(const QNSectorSet &, const std::vector<Index> &);

QN CalcDiv(const std::vector<QNSector> &, const std::vector<Index> &);

std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &);

long MulToEnd(const std::vector<long> &, int);

std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &);
} /* gqten */ 
#endif /* ifndef GQTEN_UTILS_H */
