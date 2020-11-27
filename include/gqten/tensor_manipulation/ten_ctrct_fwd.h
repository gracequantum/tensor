// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-17 13:21
* 
* Description: GraceQ/tensor project. Forward declarations for implementing tensor contraction.
*/
#ifndef GQTEN_MANIPULATION_TEN_CTRCT_FWD_H
#define GQTEN_MANIPULATION_TEN_CTRCT_FWD_H


#include <vector>

#include "gqten/fwd_dcl.h"


namespace gqten {


template <typename TenElemType>
void InitCtrctedTen(
    const GQTensor<TenElemType> *, const GQTensor<TenElemType> *,
    const std::vector<long> &, const std::vector<long> &,
    GQTensor<TenElemType> *);

template <typename TenElemType>
void WrapCtrctBlocks(
    std::vector<QNBlock<TenElemType> *> &,
    GQTensor<TenElemType> *);


template <typename TenElemType>
std::vector<QNBlock<TenElemType> *> MergeCtrctBlks(
    const std::vector<QNBlock<TenElemType> *> &);

template <typename TenElemType>
std::vector<const QNSector *> GetPNewBlkQNScts(
    const QNBlock<TenElemType> *, const QNBlock<TenElemType> *,
    const std::vector<long> &, const std::vector<long> &);

template <typename TenElemType>
bool CtrctTransChecker(
    const std::vector<long> &,
    const long,
    const char,
    std::vector<long> &);

template <typename TenElemType>
std::vector<std::size_t> GenBlksPartHashTable(
    const std::vector<QNBlock<TenElemType> *> &, const std::vector<long> &);
} /* gqten */ 
#endif /* ifndef GQTEN_MANIPULATION_TEN_CTRCT_FWD_H */
