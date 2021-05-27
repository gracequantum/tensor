// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-03-25 09:01
*
* Description: GraceQ/tensor project. Unittests for tensor expansion.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_expand.h"

#include "gtest/gtest.h"


using namespace gqten;
using U1QN = QN<U1QNVal>;
using QNSctT = QNSector<U1QN>;
using IndexT = Index<U1QN>;
using DGQTensor = GQTensor<GQTEN_Double, U1QN>;

std::string qn_nm = "qn";
U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
QNSctT qnsctp1 = QNSctT(qnp1, 1);
QNSctT qnsctp2 = QNSctT(qnp1, 2);
QNSctT qnsctm1 = QNSctT(qnm1, 1);
QNSctT qnsctm2 = QNSctT(qnm1, 2);
IndexT idx_inm1 = IndexT({qnsctm1}, IN);
IndexT idx_inp1 = IndexT({qnsctp1}, IN);
IndexT idx_outm1 = IndexT({qnsctm1}, OUT);
IndexT idx_outm2 = IndexT({qnsctm2}, OUT);
IndexT idx_outp1 = IndexT({qnsctp1}, OUT);
IndexT idx_in2 = IndexT({qnsctm1, qnsctp1}, IN);
IndexT idx_in4 = IndexT({qnsctm2, qnsctp2}, IN);
IndexT idx_out2 = IndexT({qnsctm1, qnsctp1}, OUT);


template <typename TenT>
void RunTestTenExpansionCase(
    const TenT &a,
    const TenT &b,
    const std::vector<size_t> &expand_idx_nums,
    const TenT &c
) {
  TenT res;
  Expand(&a, &b, expand_idx_nums, &res);
  EXPECT_TRUE(res == c);
}

template <typename TenT>
void RunTestTenExpansionOneIndexCase(
    TenT &a,
    TenT &b,
    const size_t &expand_idx_num,
    const TenT &c
) {
  TenT res;
  Expand(&a, &b, expand_idx_num, &res);
  EXPECT_TRUE(res == c);
}



TEST(TestExpand, TestCase) {
  DGQTensor ten0 = DGQTensor({idx_inm1, idx_outm1});
  DGQTensor ten1 = DGQTensor({idx_inp1, idx_outp1});
  ten0(0, 0) = 1.0;
  ten1(0, 0) = 1.0;
  DGQTensor ten2 = DGQTensor({idx_in2, idx_out2});
  ten2(0, 0) = 1.0;
  ten2(1, 1) = 1.0;
  DGQTensor ten3 = DGQTensor({idx_inm1, idx_outm1});
  ten3({0, 0}) = 1.0;
  DGQTensor ten4 = DGQTensor({idx_inm1, idx_outm2});
  ten4({0, 0}) = 1.0;
  ten4({0, 1}) = 1.0;
  DGQTensor ten5 = DGQTensor({idx_in4, idx_out2});
  ten5({0, 0}) = 1.0;
  ten5({1, 0}) = 1.0;
  ten5({2, 1}) = 1.0;
  ten5({3, 1}) = 1.0;

  RunTestTenExpansionCase(ten0, ten1, {0, 1}, ten2);
  RunTestTenExpansionCase(ten3, ten3, {1}, ten4);
  RunTestTenExpansionCase(ten2, ten2, {0}, ten5);

  RunTestTenExpansionOneIndexCase(ten3, ten3, 1, ten4);
  RunTestTenExpansionOneIndexCase(ten2, ten2, 0, ten5);
}
