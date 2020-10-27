// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-12 20:58
* 
* Description: GraceQ/tensor project. Type definitions used by this library.
*/
#ifndef GQTEN_FRAMEWORK_VALUE_T_H
#define GQTEN_FRAMEWORK_VALUE_T_H


#include <complex>

#include "gqten/fwd_dcl.h"


namespace gqten {


using GQTEN_Double = double;
using GQTEN_Complex = std::complex<GQTEN_Double>;


using DGQTensor = GQTensor<GQTEN_Double>;
using ZGQTensor = GQTensor<GQTEN_Complex>;
} /* gqten */ 


#define MKL_Complex16 gqten::GQTEN_Complex


#endif /* ifndef GQTEN_FRAMEWORK_VALUE_T_H */
