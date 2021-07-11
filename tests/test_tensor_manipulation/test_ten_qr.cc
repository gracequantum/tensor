// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-07-09 11:11
*
* Description: GraceQ/tensor project. Unittests for tensor QR.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_qr.h"    // QR
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract

#include "gtest/gtest.h"


using namespace gqten;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;


template <typename TenT>
void CheckIsIdTen(const TenT &t) {
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    GQTEN_Complex elem = t.GetElem({i, i});
    EXPECT_NEAR(elem.real(), 1.0, 1E-14);
    EXPECT_NEAR(elem.imag(), 0.0, 1E-14);
  }
}


template <typename TenT>
void CheckTwoTenClose(const TenT &t1, const TenT &t2) {
  EXPECT_EQ(t1.GetIndexes(), t2.GetIndexes());
  for (auto &coors : GenAllCoors(t1.GetShape())) {
    GQTEN_Complex elem1 = t1.GetElem(coors);
    GQTEN_Complex elem2 = t2.GetElem(coors);
    EXPECT_NEAR(elem1.real(), elem2.real(), 1E-14);
    EXPECT_NEAR(elem1.imag(), elem2.imag(), 1E-14);
  }
}


struct TestQr : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});

  size_t d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_4d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_4d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};


template <typename TenElemT, typename QNT>
void RunTestQrCase(
    GQTensor<TenElemT, QNT> &t,
    const size_t ldims,
    const QNT *random_div = nullptr
) {
  if (random_div != nullptr) {
    srand(0);
    t.Random(*random_div);
  }
  GQTensor<TenElemT, QNT> q, r;
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});

  QR(&t, ldims, qn0, &q, &r);

  GQTensor<TenElemT, QNT> temp;
  auto q_dag = Dag(q);
  std::vector<size_t> cano_check_u_ctrct_axes;
  for (size_t i = 0; i < ldims; ++i) { cano_check_u_ctrct_axes.push_back(i); }
  Contract(
      &q, &q_dag,
      {cano_check_u_ctrct_axes, cano_check_u_ctrct_axes},
      &temp
  );
  CheckIsIdTen(temp);

  GQTensor<TenElemT, QNT> t_restored;
  gqten::Contract(&q, &r, {{ldims}, {0}}, &t_restored);
  CheckTwoTenClose(t_restored, t);
  mkl_free_buffers();
}


TEST_F(TestQr, 2DCase) {
  RunTestQrCase(dten_2d_s, 1, &qn0);
  RunTestQrCase(dten_2d_s, 1, &qnp1);
  RunTestQrCase(dten_2d_s, 1, &qnm1);
  RunTestQrCase(dten_2d_s, 1, &qnp2);
  RunTestQrCase(dten_2d_s, 1, &qnm2);

  RunTestQrCase(zten_2d_s, 1, &qn0);
  RunTestQrCase(zten_2d_s, 1, &qnp1);
  RunTestQrCase(zten_2d_s, 1, &qnm1);
  RunTestQrCase(zten_2d_s, 1, &qnp2);
  RunTestQrCase(zten_2d_s, 1, &qnm2);
}


TEST_F(TestQr, 3DCase) {
  RunTestQrCase(dten_3d_s, 1, &qn0);
  RunTestQrCase(dten_3d_s, 1, &qnp1);
  RunTestQrCase(dten_3d_s, 1, &qnp2);
  RunTestQrCase(dten_3d_s, 1, &qnm1);
  RunTestQrCase(dten_3d_s, 1, &qnm2);
  RunTestQrCase(dten_3d_s, 2, &qn0);
  RunTestQrCase(dten_3d_s, 2, &qnp1);
  RunTestQrCase(dten_3d_s, 2, &qnp2);
  RunTestQrCase(dten_3d_s, 2, &qnm1);
  RunTestQrCase(dten_3d_s, 2, &qnm2);

  RunTestQrCase(zten_3d_s, 1, &qn0);
  RunTestQrCase(zten_3d_s, 1, &qnp1);
  RunTestQrCase(zten_3d_s, 1, &qnp2);
  RunTestQrCase(zten_3d_s, 1, &qnm1);
  RunTestQrCase(zten_3d_s, 1, &qnm2);
  RunTestQrCase(zten_3d_s, 2, &qn0);
  RunTestQrCase(zten_3d_s, 2, &qnp1);
  RunTestQrCase(zten_3d_s, 2, &qnp2);
  RunTestQrCase(zten_3d_s, 2, &qnm1);
  RunTestQrCase(zten_3d_s, 2, &qnm2);
}


TEST_F(TestQr, 4DCase) {
  RunTestQrCase(dten_4d_s, 1, &qn0);
  RunTestQrCase(dten_4d_s, 1, &qnp1);
  RunTestQrCase(dten_4d_s, 1, &qnp2);
  RunTestQrCase(dten_4d_s, 1, &qnm1);
  RunTestQrCase(dten_4d_s, 1, &qnm2);
  RunTestQrCase(dten_4d_s, 2, &qn0);
  RunTestQrCase(dten_4d_s, 2, &qnp1);
  RunTestQrCase(dten_4d_s, 2, &qnp2);
  RunTestQrCase(dten_4d_s, 2, &qnm1);
  RunTestQrCase(dten_4d_s, 2, &qnm2);
  RunTestQrCase(dten_4d_s, 3, &qn0);
  RunTestQrCase(dten_4d_s, 3, &qnp1);
  RunTestQrCase(dten_4d_s, 3, &qnp2);
  RunTestQrCase(dten_4d_s, 3, &qnm1);
  RunTestQrCase(dten_4d_s, 3, &qnm2);

  RunTestQrCase(zten_4d_s, 1, &qn0);
  RunTestQrCase(zten_4d_s, 1, &qnp1);
  RunTestQrCase(zten_4d_s, 1, &qnp2);
  RunTestQrCase(zten_4d_s, 1, &qnm1);
  RunTestQrCase(zten_4d_s, 1, &qnm2);
  RunTestQrCase(zten_4d_s, 2, &qn0);
  RunTestQrCase(zten_4d_s, 2, &qnp1);
  RunTestQrCase(zten_4d_s, 2, &qnp2);
  RunTestQrCase(zten_4d_s, 2, &qnm1);
  RunTestQrCase(zten_4d_s, 2, &qnm2);
  RunTestQrCase(zten_4d_s, 3, &qn0);
  RunTestQrCase(zten_4d_s, 3, &qnp1);
  RunTestQrCase(zten_4d_s, 3, &qnp2);
  RunTestQrCase(zten_4d_s, 3, &qnm1);
  RunTestQrCase(zten_4d_s, 3, &qnm2);
}
