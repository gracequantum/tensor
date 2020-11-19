// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-12 20:58
* 
* Description: GraceQ/tensor project. Type definitions used by this library.
* This file must be included before any MKL library header file.
*/
#ifndef GQTEN_FRAMEWORK_VALUE_T_H
#define GQTEN_FRAMEWORK_VALUE_T_H


#include <complex>    // complex
#include <vector>     // vector


namespace gqten {


using GQTEN_Double = double;
using GQTEN_Complex = std::complex<GQTEN_Double>;


using CoorsT = std::vector<size_t>;
using ShapeT = std::vector<size_t>;
} /* gqten */


#define MKL_Complex16 gqten::GQTEN_Complex    // This must be defined before any MKL header file.


#endif /* ifndef GQTEN_FRAMEWORK_VALUE_T_H */
