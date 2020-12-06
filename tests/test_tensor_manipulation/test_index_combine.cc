// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-06 17:16
*
* Description: GraceQ/tensor project. Unittests for index combination.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/index_combine.h"
#include "gqten/tensor_manipulation/basic_operations.h"
#include "gqten/tensor_manipulation/ten_ctrct.h"

#include "gtest/gtest.h"


using namespace gqten;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;

std::string qn_nm = "qn";
U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});
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
IndexT idx_out2 = IndexT({qnsctm1, qnsctp1}, OUT);


template <typename CombinerElemT, typename QNT>
void TestCombinerIntraStruct(const GQTensor<CombinerElemT, QNT> &combiner) {
  auto combiner_dag = Dag(combiner);
  GQTensor<CombinerElemT, QNT> res;
  Contract(&combiner, &combiner_dag, {{0, 1}, {0, 1}}, &res);
  EXPECT_EQ(res.Rank(), 2);
  EXPECT_EQ(res.GetShape()[0], res.GetShape()[1]);
  EXPECT_EQ(
      res.GetShape()[0],
      combiner.GetIndexes()[0].dim() * combiner.GetIndexes()[1].dim()
  );
  for (size_t i = 0; i < res.GetShape()[0]; ++i) {
    for (size_t j = 0; j < res.GetShape()[1]; ++j) {
      if (i == j) {
        EXPECT_EQ(res.GetElem({i, j}), 1.0);
      } else {
        EXPECT_EQ(res.GetElem({i, j}), 0.0);
      }
    }
  }

  mkl_free_buffers();
}


TEST(TestIndexCombine, TestCase) {
  auto combiner = IndexCombine<GQTEN_Double, U1QN>(idx_inp1, idx_inp1, OUT);
  EXPECT_EQ(Div(combiner), qn0);
  EXPECT_EQ(combiner.GetIndexes()[0], idx_inp1);
  EXPECT_EQ(combiner.GetIndexes()[1], idx_inp1);
  TestCombinerIntraStruct(combiner);

  combiner = IndexCombine<GQTEN_Double, U1QN>(idx_inp1, idx_inp1, IN);
  EXPECT_EQ(Div(combiner), qn0);
  EXPECT_EQ(combiner.GetIndexes()[0], idx_inp1);
  EXPECT_EQ(combiner.GetIndexes()[1], idx_inp1);
  TestCombinerIntraStruct(combiner);

  combiner = IndexCombine<GQTEN_Double, U1QN>(idx_in2, idx_in2);
  EXPECT_EQ(combiner.GetIndexes()[0], idx_in2);
  EXPECT_EQ(combiner.GetIndexes()[1], idx_in2);
  TestCombinerIntraStruct(combiner);

  combiner = IndexCombine<GQTEN_Double, U1QN>(idx_in2, idx_in2, IN);
  EXPECT_EQ(combiner.GetIndexes()[0], idx_in2);
  EXPECT_EQ(combiner.GetIndexes()[1], idx_in2);
  TestCombinerIntraStruct(combiner);
}
