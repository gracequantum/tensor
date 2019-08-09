// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-09 11:37
* 
* Description: GraceQ/tensor project. Intra-used functions for tensor linear combination.
*/
#ifndef GQTEN_TEN_LINCMB_H
#define GQTEN_TEN_LINCMB_H


#include "gqten/gqten.h"


namespace gqten {


void LinearCombineOneTerm(const double, const GQTensor *, GQTensor *);
} /* gqten */ 
#endif /* ifndef GQTEN_TEN_LINCMB_H */
