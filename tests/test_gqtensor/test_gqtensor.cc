// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-17 15:19
*
* Description: GraceQ/tensor project. Unittests for GQTensor object.
*/
#include "gqten/gqtensor/gqtensor.h"        // GQTensor
#include "gqten/gqtensor/index.h"           // Index
#include "gqten/gqtensor/qn/qn.h"           // QN
#include "gqten/gqtensor/qn/qnval_u1.h"     // U1QNVal
#include "gqten/gqtensor/qnsct.h"           // QNSectorVec
#include "gqten/utility/utils_inl.h"        // GenAllCoors

#include "gtest/gtest.h"
#include "../testing_utility.h"     // RandInt, RandUnsignedInt, TransCoors


using namespace gqten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;


struct TestGQTensor : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  QNSctT qnsct0_s =  QNSctT(qn0,  4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
  QNSctT qnsct0_l =  QNSctT(qn0,  10);
  QNSctT qnsctp1_l = QNSctT(qnp1, 8);
  QNSctT qnsctm1_l = QNSctT(qnm1, 12);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::OUT);
  IndexT idx_in_l =  IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::OUT);

  DGQTensor dten_default = DGQTensor();
  DGQTensor dten_scalar = DGQTensor(IndexVec<U1QN>{});
  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_1d_l = DGQTensor({idx_out_l});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_2d_l = DGQTensor({idx_in_l, idx_out_l});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_3d_l = DGQTensor({idx_in_l, idx_out_l, idx_out_l});
  ZGQTensor zten_default = ZGQTensor();
  DGQTensor zten_scalar = DGQTensor(IndexVec<U1QN>{});
  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_1d_l = ZGQTensor({idx_out_l});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_2d_l = ZGQTensor({idx_in_l, idx_out_l});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_3d_l = ZGQTensor({idx_in_l, idx_out_l, idx_out_l});
};


template <typename GQTensorT>
void RunTestGQTensorCommonConstructorCase(
    const GQTensorT &ten,
    const std::vector<IndexT> &indexes) {
  EXPECT_EQ(ten.GetIndexes(), indexes);

  size_t size = 1;
  for (size_t i = 0; i < ten.Rank(); i++) {
    auto dim = ten.GetShape()[i];
    EXPECT_EQ(dim, indexes[i].dim());
    size *= dim;
  }

  if (ten.IsDefault()) {
    EXPECT_EQ(ten.size(), 0);
  } else {
    EXPECT_EQ(ten.size(), size);
  }
}


TEST_F(TestGQTensor, TestCommonConstructor) {
  EXPECT_TRUE(dten_default.IsDefault());
  RunTestGQTensorCommonConstructorCase(dten_scalar, {});
  RunTestGQTensorCommonConstructorCase(dten_1d_s, {idx_out_s});
  RunTestGQTensorCommonConstructorCase(dten_2d_s, {idx_in_s, idx_out_s});
  RunTestGQTensorCommonConstructorCase(
      dten_3d_s,
      {idx_in_s, idx_out_s, idx_out_s});

  EXPECT_TRUE(zten_default.IsDefault());
  RunTestGQTensorCommonConstructorCase(zten_scalar, {});
  RunTestGQTensorCommonConstructorCase(zten_1d_s, {idx_out_s});
  RunTestGQTensorCommonConstructorCase(zten_2d_s, {idx_in_s, idx_out_s});
  RunTestGQTensorCommonConstructorCase(
      zten_3d_s,
      {idx_in_s, idx_out_s, idx_out_s});
}


template <typename ElemT, typename QNT>
void RunTestGQTensorElemAssignmentCase(
    const GQTensor<ElemT, QNT> &t_init,
    const std::vector<ElemT> elems,
    const std::vector<std::vector<size_t>> coors) {
  auto t = t_init;
  for (size_t i = 0; i < elems.size(); ++i) {
    t.SetElem(coors[i], elems[i]);
  }
  for (auto coor : GenAllCoors(t.GetShape())) {
    auto coor_it = std::find(coors.cbegin(), coors.cend(), coor);
    if (coor_it != coors.end()) {
      auto elem_idx = std::distance(coors.cbegin(), coor_it);
      EXPECT_EQ(t.GetElem(coor), elems[elem_idx]);
    } else {
      EXPECT_EQ(t.GetElem(coor), ElemT(0.0));
    }
  }
}


TEST_F(TestGQTensor, TestElemAssignment) {
  RunTestGQTensorElemAssignmentCase(dten_scalar, {drand()}, {{}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand()}, {{0}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand()}, {{1}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand(), drand()}, {{0}, {1}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand(), drand()}, {{1}, {2}});
  RunTestGQTensorElemAssignmentCase(dten_2d_s, {drand()}, {{0, 0}});
  RunTestGQTensorElemAssignmentCase(dten_2d_s, {drand()}, {{2, 3}});
  RunTestGQTensorElemAssignmentCase(
      dten_2d_s,
      {drand(), drand()},
      {{2, 3}, {1, 7}});
  RunTestGQTensorElemAssignmentCase(dten_3d_s, {drand()}, {{0, 0, 0}});
  RunTestGQTensorElemAssignmentCase(dten_3d_s, {drand()}, {{2, 3, 4}});
  RunTestGQTensorElemAssignmentCase(
      dten_3d_s,
      {drand(), drand()},
      {{2, 3, 5}, {1, 7, 4}});
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand()},
      {{0}});
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand()},
      {{1}});
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand(), zrand()},
      {{0}, {1}});
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand(), zrand()},
      {{1}, {2}});
  RunTestGQTensorElemAssignmentCase(
      zten_2d_s,
      {zrand()},
      {{0, 0}});
  RunTestGQTensorElemAssignmentCase(
      zten_2d_s,
      {zrand()},
      {{2, 3}});
  RunTestGQTensorElemAssignmentCase(
      zten_2d_s,
      {zrand(), zrand()},
      {{2, 3}, {1, 7}});
  RunTestGQTensorElemAssignmentCase(zten_3d_s, {zrand()}, {{0, 0, 0}});
  RunTestGQTensorElemAssignmentCase(zten_3d_s, {zrand()}, {{2, 3, 4}});
  RunTestGQTensorElemAssignmentCase(
      zten_3d_s,
      {zrand(), zrand()},
      {{2, 3, 5}, {1, 7, 4}});
}


template <typename ElemT, typename QNT>
void RunTestGQTensorRandomCase(
    GQTensor<ElemT, QNT> &t,
    const QNT &div,
    const std::vector<std::vector<QNSector<QNT>>> &qnscts_set) {
  std::vector<std::vector<QNSector<QNT>>> had_qnscts_set;
  srand(0);
  t.Random(div);

  EXPECT_EQ(t.GetQNBlkNum(), qnscts_set.size());
  EXPECT_EQ(t.Div(), div);

  if (t.IsScalar()) {
    srand(0);
    EXPECT_EQ(t.GetElem({}), RandT<ElemT>());
  }
  // TODO: Check each element in the random tensor.
}


TEST_F(TestGQTensor, Random) {
  RunTestGQTensorRandomCase(dten_scalar, U1QN(), {});
  RunTestGQTensorRandomCase(dten_1d_s, qn0, {{qnsct0_s}});
  RunTestGQTensorRandomCase(dten_1d_s, qnp1, {{qnsctp1_s}});
  RunTestGQTensorRandomCase(dten_1d_l, qn0, {{qnsct0_l}});
  RunTestGQTensorRandomCase(dten_1d_l, qnp1, {{qnsctp1_l}});
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s},
        {qnsctp1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qnm1,
      {
        {qnsct0_s, qnsctm1_s},
        {qnsctp1_s, qnsct0_s}
      });
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l},
        {qnsctp1_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qnm1,
      {
        {qnsct0_l, qnsctm1_l},
        {qnsctp1_l, qnsct0_l}
      });
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      dten_3d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctm1_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsct0_s},
        {qnsctp1_s, qnsct0_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      dten_3d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s, qnsct0_s},
        {qnsctm1_s, qnsctm1_s, qnsctp1_s},
        {qnsctm1_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctp1_s, qnsct0_s},
        {qnsct0_s, qnsct0_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      dten_3d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctp1_s},
        {qnsct0_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      dten_3d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctm1_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsct0_l},
        {qnsctp1_l, qnsct0_l, qnsctp1_l} 
      });
  RunTestGQTensorRandomCase(
      dten_3d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l, qnsct0_l},
        {qnsctm1_l, qnsctm1_l, qnsctp1_l},
        {qnsctm1_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctp1_l, qnsct0_l},
        {qnsct0_l, qnsct0_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      dten_3d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctp1_l},
        {qnsct0_l, qnsctp1_l, qnsctp1_l}
      });

  RunTestGQTensorRandomCase(zten_scalar, U1QN(), {});
  RunTestGQTensorRandomCase(zten_1d_s, qn0, {{qnsct0_s}});
  RunTestGQTensorRandomCase(zten_1d_s, qnp1, {{qnsctp1_s}});
  RunTestGQTensorRandomCase(zten_1d_l, qn0, {{qnsct0_l}});
  RunTestGQTensorRandomCase(zten_1d_l, qnp1, {{qnsctp1_l}});
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s},
        {qnsctp1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qnm1,
      {
        {qnsct0_s, qnsctm1_s},
        {qnsctp1_s, qnsct0_s}
      });
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l},
        {qnsctp1_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qnm1,
      {
        {qnsct0_l, qnsctm1_l},
        {qnsctp1_l, qnsct0_l}
      });
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      zten_3d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctm1_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsct0_s},
        {qnsctp1_s, qnsct0_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      zten_3d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s, qnsct0_s},
        {qnsctm1_s, qnsctm1_s, qnsctp1_s},
        {qnsctm1_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctp1_s, qnsct0_s},
        {qnsct0_s, qnsct0_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      zten_3d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctp1_s},
        {qnsct0_s, qnsctp1_s, qnsctp1_s}
      });
  RunTestGQTensorRandomCase(
      zten_3d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctm1_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsct0_l},
        {qnsctp1_l, qnsct0_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      zten_3d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l, qnsct0_l},
        {qnsctm1_l, qnsctm1_l, qnsctp1_l},
        {qnsctm1_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctp1_l, qnsct0_l},
        {qnsct0_l, qnsct0_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsctp1_l}
      });
  RunTestGQTensorRandomCase(
      zten_3d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctp1_l},
        {qnsct0_l, qnsctp1_l, qnsctp1_l}
      });
}


template <typename GQTensorT>
void RunTestGQTensorEqCase(
    const GQTensorT &lhs, const GQTensorT &rhs, bool test_eq = true) {
  if (test_eq) {
    EXPECT_TRUE(lhs == rhs);
  } else {
    EXPECT_TRUE(lhs != rhs);
  }
}


TEST_F(TestGQTensor, TestEq) {
  RunTestGQTensorEqCase(dten_default, dten_default);

  RunTestGQTensorEqCase(dten_scalar, dten_scalar);
  dten_scalar.Random(U1QN());
  decltype(dten_scalar) dten_scalar2(dten_scalar.GetIndexes());
  RunTestGQTensorEqCase(dten_scalar, dten_scalar2, false);

  RunTestGQTensorEqCase(dten_1d_s, dten_1d_s);
  dten_1d_s.Random(qn0);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_s);
  decltype(dten_1d_s) dten_1d_s2(dten_1d_s.GetIndexes());
  dten_1d_s2.Random(qnp1);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_s2, false);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_l, false);
  RunTestGQTensorEqCase(dten_1d_s, dten_2d_s, false);

  RunTestGQTensorEqCase(zten_default, zten_default);

  RunTestGQTensorEqCase(zten_scalar, zten_scalar);
  zten_scalar.Random(U1QN());
  decltype(zten_scalar) zten_scalar2(zten_scalar.GetIndexes());
  RunTestGQTensorEqCase(zten_scalar, zten_scalar2, false);

  RunTestGQTensorEqCase(zten_1d_s, zten_1d_s);
  zten_1d_s.Random(qn0);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_s);
  decltype(zten_1d_s) zten_1d_s2(zten_1d_s.GetIndexes());
  zten_1d_s2.Random(qnp1);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_s2, false);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_l, false);
  RunTestGQTensorEqCase(zten_1d_s, zten_2d_s, false);
}


template <typename GQTensorT>
void RunTestGQTensorCopyAndMoveConstructorsCase(const GQTensorT &t) {
  GQTensorT gqten_cpy(t);
  EXPECT_EQ(gqten_cpy, t);
  auto gqten_cpy2 = t;
  EXPECT_EQ(gqten_cpy2, t);

  GQTensorT gqten_tomove(t);    // Copy it.
  GQTensorT gqten_moved(std::move(gqten_tomove));
  EXPECT_EQ(gqten_moved, t);
  EXPECT_EQ(gqten_tomove.GetBlkSparDataTen(), nullptr);
  GQTensorT gqten_tomove2(t);
  auto gqten_moved2 = std::move(gqten_tomove2);
  EXPECT_EQ(gqten_moved2, t);
  EXPECT_EQ(gqten_tomove2.GetBlkSparDataTen(), nullptr);
}


TEST_F(TestGQTensor, TestCopyAndMoveConstructors) {
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_3d_s);

  RunTestGQTensorCopyAndMoveConstructorsCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(zten_3d_s);
}


template <typename GQTensorT>
void RunTestGQTensorTransposeCase(
    const GQTensorT &t, const std::vector<size_t> &axes) {
  auto transed_t = t;
  transed_t.Transpose(axes);
  if (t.IsDefault() || t.IsScalar()) {
    EXPECT_EQ(transed_t, t);
  } else {
    for (size_t i = 0; i < axes.size(); ++i) {
      EXPECT_EQ(transed_t.GetIndexes()[i], t.GetIndexes()[axes[i]]);
      EXPECT_EQ(transed_t.GetShape()[i], t.GetShape()[axes[i]]);
    }
    for (auto &coors : GenAllCoors(t.GetShape())) {
      EXPECT_EQ(
          transed_t.GetElem(TransCoors(coors, axes)),
          t.GetElem(coors)
      );
    }
  }
}


TEST_F(TestGQTensor, TestTranspose) {
  RunTestGQTensorTransposeCase(dten_default, {});
  dten_1d_s.Random(qn0);
  RunTestGQTensorTransposeCase(dten_1d_s, {0});
  dten_2d_s.Random(qn0);
  RunTestGQTensorTransposeCase(dten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(dten_2d_s, {1, 0});
  dten_2d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(dten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(dten_2d_s, {1, 0});
  dten_3d_s.Random(qn0);
  RunTestGQTensorTransposeCase(dten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {2, 0, 1});
  dten_3d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(dten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {2, 0, 1});

  RunTestGQTensorTransposeCase(zten_default, {});
  zten_1d_s.Random(qn0);
  RunTestGQTensorTransposeCase(zten_1d_s, {0});
  zten_2d_s.Random(qn0);
  RunTestGQTensorTransposeCase(zten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(zten_2d_s, {1, 0});
  zten_2d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(zten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(zten_2d_s, {1, 0});
  zten_3d_s.Random(qn0);
  RunTestGQTensorTransposeCase(zten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {2, 0, 1});
  zten_3d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(zten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {2, 0, 1});
}


template <typename GQTensorT>
void RunTestGQTensorNormalizeCase(GQTensorT &t) {
  auto norm2 = 0.0;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    norm2 += std::norm(t.GetElem(coors));
  }
  auto norm = t.Normalize();
  EXPECT_DOUBLE_EQ(norm, std::sqrt(norm2));

  norm2 = 0.0;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    norm2 += std::norm(t.GetElem(coors));
  }
  EXPECT_NEAR(norm2, 1.0, kEpsilon);
}


TEST_F(TestGQTensor, TestNormalize) {
  dten_scalar.Random(U1QN());
  auto dscalar = dten_scalar.GetElem({});
  auto dnorm = dten_scalar.Normalize();
  EXPECT_DOUBLE_EQ(dnorm, std::abs(dscalar));
  EXPECT_DOUBLE_EQ(dten_scalar.GetElem({}), 1.0);

  dten_1d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(dten_1d_s);
  dten_1d_s.Random(qnp1);
  RunTestGQTensorNormalizeCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(dten_3d_s);

  zten_scalar.Random(U1QN());
  auto zscalar = zten_scalar.GetElem({});
  auto znorm = zten_scalar.Normalize();
  EXPECT_DOUBLE_EQ(znorm, std::abs(zscalar));
  EXPECT_DOUBLE_EQ(zten_scalar.GetElem({}), 1.0);

  zten_1d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(zten_1d_s);
  zten_1d_s.Random(qnp1);
  RunTestGQTensorNormalizeCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(zten_3d_s);
}


template <typename GQTensorT>
void RunTestGQTensorDagCase(const GQTensorT &t) {
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


TEST_F(TestGQTensor, TestDag) {
  RunTestGQTensorDagCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestGQTensorDagCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorDagCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorDagCase(dten_3d_s);

  RunTestGQTensorDagCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestGQTensorDagCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorDagCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorDagCase(zten_3d_s);
}
