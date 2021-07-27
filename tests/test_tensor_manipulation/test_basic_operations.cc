// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-27 09:33
*
* Description: GraceQ/tensor project. Unittests for basic tensor operations.
*/
#include "gqten/gqtensor_all.h"      // GQTensor, Index, QN, U1QNVal, QNSectorVec
#include "gqten/tensor_manipulation/basic_operations.h"

#include "gtest/gtest.h"

using namespace gqten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;


struct TestBasicTensorOperations : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  QNSctT qnsct0_s =  QNSctT(qn0,  4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::OUT);

  DGQTensor dten_default = DGQTensor();
  DGQTensor dten_scalar = DGQTensor(IndexVec<U1QN>{});
  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_default = ZGQTensor();
  ZGQTensor zten_scalar = ZGQTensor(IndexVec<U1QN>{});
  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
};


template <typename GQTensorT>
void RunTestTensorDagCase(const GQTensorT &t) {
  if (t.IsDefault()) {
    // Do nothing
  } else if (t.IsScalar()) {
    auto t_dag = Dag(t);
    EXPECT_EQ(t_dag.GetElem({}), std::conj(t.GetElem({})));
  } else {
    auto t_dag = Dag(t);
    for (size_t i = 0; i < t.Rank(); ++i) {
      EXPECT_EQ(t_dag.GetIndexes()[i], InverseIndex(t.GetIndexes()[i]));
    }
    for (auto &coor : GenAllCoors(t.GetShape())) {
      EXPECT_EQ(t_dag.GetElem(coor), std::conj(t.GetElem(coor)));
    }
  }
}


TEST_F(TestBasicTensorOperations, TestDag) {
  RunTestTensorDagCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestTensorDagCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestTensorDagCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestTensorDagCase(dten_3d_s);

  RunTestTensorDagCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestTensorDagCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestTensorDagCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestTensorDagCase(zten_3d_s);
}


template <typename GQTensorT>
void RunTestTensorConjCase(const GQTensorT &t) {
  if (t.IsDefault()) {
    // Do nothing
  } else if (t.IsScalar()) {
    auto t_conj = Conj(t);
    EXPECT_EQ(t_conj.GetElem({}), std::conj(t.GetElem({})));
  } else {
    auto t_conj = Conj(t);
    for (size_t i = 0; i < t.Rank(); ++i) {
      EXPECT_EQ(t_conj.GetIndexes()[i], t.GetIndexes()[i]);
    }
    for (auto &coor : GenAllCoors(t.GetShape())) {
      EXPECT_EQ(t_conj.GetElem(coor), std::conj(t.GetElem(coor)));
    }
  }
}


TEST_F(TestBasicTensorOperations, TestConj) {
  RunTestTensorConjCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestTensorConjCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestTensorConjCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestTensorConjCase(dten_3d_s);

  RunTestTensorConjCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestTensorConjCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestTensorConjCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestTensorConjCase(zten_3d_s);
}


template <typename ElemT, typename QNT>
void RunTestTensorDivCase(GQTensor<ElemT, QNT> &t, const QNT &div) {
  t.Random(div);
  EXPECT_EQ(Div(t), div);
}


TEST_F(TestBasicTensorOperations, TestDiv) {
  dten_scalar.Random(U1QN());
  EXPECT_EQ(Div(dten_scalar), U1QN());
  RunTestTensorDivCase(dten_1d_s, qn0);
  RunTestTensorDivCase(dten_2d_s, qn0);
  RunTestTensorDivCase(dten_2d_s, qnp1);
  RunTestTensorDivCase(dten_2d_s, qnm1);
  RunTestTensorDivCase(dten_2d_s, qnp2);
  RunTestTensorDivCase(dten_3d_s, qn0);
  RunTestTensorDivCase(dten_3d_s, qnp1);
  RunTestTensorDivCase(dten_3d_s, qnm1);
  RunTestTensorDivCase(dten_3d_s, qnp2);

  zten_scalar.Random(U1QN());
  EXPECT_EQ(Div(zten_scalar), U1QN());
  RunTestTensorDivCase(zten_1d_s, qn0);
  RunTestTensorDivCase(zten_2d_s, qn0);
  RunTestTensorDivCase(zten_2d_s, qnp1);
  RunTestTensorDivCase(zten_2d_s, qnm1);
  RunTestTensorDivCase(zten_2d_s, qnp2);
  RunTestTensorDivCase(zten_3d_s, qn0);
  RunTestTensorDivCase(zten_3d_s, qnp1);
  RunTestTensorDivCase(zten_3d_s, qnm1);
  RunTestTensorDivCase(zten_3d_s, qnp2);
}


template <typename QNT>
void RunTestRealTensorToComplexCase(
    GQTensor<GQTEN_Double, QNT> &t,
    const QNT & div,
    unsigned int rand_seed) {
  srand(rand_seed);
  t.Random(div);
  auto zten = ToComplex(t);
  for (auto &coors : GenAllCoors(t.GetShape())) {
    EXPECT_DOUBLE_EQ(zten.GetElem(coors).real(), t.GetElem(coors));
    EXPECT_DOUBLE_EQ(zten.GetElem(coors).imag(), 0.0);
  }
}


TEST_F(TestBasicTensorOperations, TestToComplex) {
  dten_scalar.Random(U1QN());
  auto zten = ToComplex(dten_scalar);
  EXPECT_DOUBLE_EQ(zten.GetElem({}).real(), dten_scalar.GetElem({}));
  EXPECT_DOUBLE_EQ(zten.GetElem({}).imag(), 0.0);

  RunTestRealTensorToComplexCase(dten_1d_s, qn0, 0);
  RunTestRealTensorToComplexCase(dten_1d_s, qn0, 1);
  RunTestRealTensorToComplexCase(dten_1d_s, qnp1, 0);
  RunTestRealTensorToComplexCase(dten_1d_s, qnp1, 1);
  RunTestRealTensorToComplexCase(dten_2d_s, qn0, 0);
  RunTestRealTensorToComplexCase(dten_2d_s, qn0, 1);
  RunTestRealTensorToComplexCase(dten_2d_s, qnp1, 0);
  RunTestRealTensorToComplexCase(dten_2d_s, qnp1, 1);
  RunTestRealTensorToComplexCase(dten_3d_s, qn0, 0);
  RunTestRealTensorToComplexCase(dten_3d_s, qn0, 1);
  RunTestRealTensorToComplexCase(dten_3d_s, qnp1, 0);
  RunTestRealTensorToComplexCase(dten_3d_s, qnp1, 1);
}
