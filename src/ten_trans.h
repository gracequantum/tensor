// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 19:02
* 
* Description: GraceQ/tensor project. Dense tensor transpose functions.
*/
#ifndef GQTEN_TEN_TRANS_H
#define GQTEN_TEN_TRANS_H


#include <vector>


namespace gqten {


double *DenseTensorTranspose(
    const double *,
    const long,
    const long,
    const std::vector<long> &,
    const std::vector<long> &);
} /* gqten */ 
#endif /* ifndef GQTEN_TEN_TRANS_H */
