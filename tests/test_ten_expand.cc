// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-06-22 09:51
* 
* Description: GraceQ/tensor project. Unittests for tensor expansion function.
*/
#include "gqten/gqten.h"
#include "testing_utils.h"
#include "gtest/gtest.h"


using namespace gqten;


std::string qn_nm = "qn";
QN qn0 =  QN({QNNameVal(qn_nm,  0)});
QN qnp1 = QN({QNNameVal(qn_nm,  1)});
QN qnp2 = QN({QNNameVal(qn_nm,  2)});
QN qnm1 = QN({QNNameVal(qn_nm, -1)});
QNSector qnsctp1 = QNSector(qnp1, 1);
QNSector qnsctp2 = QNSector(qnp1, 2);
QNSector qnsctm1 = QNSector(qnm1, 1);
QNSector qnsctm2 = QNSector(qnm1, 2);
Index idx_inm1 = Index({qnsctm1}, IN);
Index idx_inp1 = Index({qnsctp1}, IN);
Index idx_outm1 = Index({qnsctm1}, OUT);
Index idx_outm2 = Index({qnsctm2}, OUT);
Index idx_outp1 = Index({qnsctp1}, OUT);
Index idx_in2 = Index({qnsctm1, qnsctp1}, IN);
Index idx_out2 = Index({qnsctm1, qnsctp1}, OUT);


template <typename TenType>
void RunTestTenExpansionCase(
    const TenType &ta,
    const TenType &tb,
    const std::vector<size_t> &expand_idx_nums,
    const TenType &tc
) {
  TenType res;
  Expand(&ta, &tb, expand_idx_nums, &res);
  EXPECT_TRUE(res == tc);
}


TEST(TestExpand, TestCase) {
  DGQTensor ten0 = DGQTensor({idx_inm1, idx_outm1});
  DGQTensor ten1 = DGQTensor({idx_inp1, idx_outp1});
  ten0({0, 0}) = 1.0;
  ten1({0, 0}) = 1.0;
  RunTestTenExpansionCase(ten0, ten0, {}, 2.0 * ten0);

  DGQTensor ten2 = DGQTensor({idx_in2, idx_out2});
  ten2({0, 0}) = 1.0;
  ten2({1, 1}) = 1.0;
  RunTestTenExpansionCase(ten0, ten1, {0, 1}, ten2);

  DGQTensor ten3 = DGQTensor({idx_inm1, idx_outm1});
  ten3({0, 0}) = 1.0;
  DGQTensor ten4 = DGQTensor({idx_inm1, idx_outm2});
  ten4({0, 0}) = 1.0;
  ten4({0, 1}) = 1.0;
  RunTestTenExpansionCase(ten3, ten3, {1}, ten4);
}
