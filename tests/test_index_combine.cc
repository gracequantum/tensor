// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-06-29 10:30
* 
* Description: GraceQ/tensor project. Unittests for index combination.
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
QN qnm2 = QN({QNNameVal(qn_nm, -2)});
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


template <typename CombinerElemT>
void TestCombinerIntraStruct(const GQTensor<CombinerElemT> &combiner) {
  auto combiner_dag = Dag(combiner);
  GQTensor<CombinerElemT> res;
  Contract(&combiner, &combiner_dag, {{0, 1}, {0, 1}}, &res);
  EXPECT_EQ(res.shape.size(), 2);
  EXPECT_EQ(res.shape[0], res.shape[1]);
  EXPECT_EQ(res.shape[0], combiner.indexes[0].dim * combiner.indexes[1].dim);
  for (int i = 0; i < res.shape[0]; ++i) {
    for (int j = 0; j < res.shape[1]; ++j) {
      if (i == j) {
        EXPECT_EQ(res.Elem({i, j}), 1.0);
      } else {
        EXPECT_EQ(res.Elem({i, j}), 0.0);
      }
    }
  }
}


TEST(TestIndexCombine, TestCase) {
  auto combiner = IndexCombine<GQTEN_Double>(idx_inp1, idx_inp1);
  EXPECT_EQ(Div(combiner), qn0);
  EXPECT_EQ(combiner.indexes[0], idx_inp1);
  EXPECT_EQ(combiner.indexes[1], idx_inp1);
  TestCombinerIntraStruct(combiner);

  combiner = IndexCombine<GQTEN_Double>(idx_inp1, idx_inp1, IN);
  EXPECT_EQ(Div(combiner), qn0);
  EXPECT_EQ(combiner.indexes[0], idx_inp1);
  EXPECT_EQ(combiner.indexes[1], idx_inp1);
  TestCombinerIntraStruct(combiner);

  combiner = IndexCombine<GQTEN_Double>(idx_in2, idx_in2);
  EXPECT_EQ(combiner.indexes[0], idx_in2);
  EXPECT_EQ(combiner.indexes[1], idx_in2);
  TestCombinerIntraStruct(combiner);

  combiner = IndexCombine<GQTEN_Double>(idx_in2, idx_in2, IN);
  EXPECT_EQ(combiner.indexes[0], idx_in2);
  EXPECT_EQ(combiner.indexes[1], idx_in2);
  TestCombinerIntraStruct(combiner);
}
