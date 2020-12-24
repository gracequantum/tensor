// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-11 09:38
*
* Description: GraceQ/tensor project. Unittests for tensor linear combination functions.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_linear_combine.h"

#include "gtest/gtest.h"


using namespace gqten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;


using namespace gqten;


struct TestLinearCombination : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  int d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DGQTensor dten_default = DGQTensor();
  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});

  ZGQTensor zten_default = ZGQTensor();
  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
};


template <typename TenElemT, typename QNT>
void RunTestLinearCombinationCase(
    const std::vector<TenElemT> &coefs,
    const std::vector<GQTensor<TenElemT, QNT> *> &pts,
    GQTensor<TenElemT, QNT> *res,
    const TenElemT beta = 0.0
) {
  GQTensor<TenElemT, QNT> bnchmrk;
  if (!(beta == TenElemT(0.0))) {
    bnchmrk = beta * (*res);
  }
  auto nt = pts.size();
  for (size_t i = 0; i < nt; ++i) {
    auto temp = coefs[i] * (*pts[i]);
    if (i == 0 && beta == TenElemT(0.0)) {
      bnchmrk = temp;
    } else {
      bnchmrk += temp;
    }
  }

  LinearCombine(coefs, pts, beta, res);

  EXPECT_EQ(*res, bnchmrk);
}


TEST_F(TestLinearCombination, 1TenCases) {
  DGQTensor dten_res;
  dten_res = dten_1d_s;
  dten_1d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res, drand());
  dten_1d_s.Random(qnp1);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res, drand());
  dten_res = dten_2d_s;
  dten_2d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res, drand());
  dten_2d_s.Random(qnm1);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res, drand());
  dten_res = dten_3d_s;
  dten_3d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res, drand());
  dten_3d_s.Random(qnp2);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res, drand());

  ZGQTensor zten_res;
  zten_res = zten_1d_s;
  zten_1d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res, zrand());
  zten_1d_s.Random(qnp1);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res, zrand());
  zten_res = zten_2d_s;
  zten_2d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res,zrand());
  zten_2d_s.Random(qnm1);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res,zrand());
  zten_res = zten_3d_s;
  zten_3d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res,zrand());
  zten_3d_s.Random(qnp2);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res,zrand());
}


TEST_F(TestLinearCombination, 2TenCases) {
  DGQTensor dten_res;
  auto dten_1d_s2 = dten_1d_s;
  auto dten_2d_s2 = dten_2d_s;
  auto dten_3d_s2 = dten_3d_s;
  dten_res = dten_1d_s;
  dten_1d_s.Random(qn0);
  dten_1d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res,
      drand()
  );
  dten_1d_s.Random(qn0);
  dten_1d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res,
      drand()
  );
  dten_1d_s.Random(qnm1);
  dten_1d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2},
      &dten_res,
      drand()
  );

  dten_res = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res,
      drand()
  );
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res,
      drand()
  );
  dten_2d_s.Random(qnm1);
  dten_2d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2},
      &dten_res,
      drand()
  );

  dten_res = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res,
      drand()
  );
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res,
      drand()
  );
  dten_3d_s.Random(qnm1);
  dten_3d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res
  );
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2},
      &dten_res,
      drand()
  );

  ZGQTensor zten_res;
  auto zten_1d_s2 = zten_1d_s;
  auto zten_2d_s2 = zten_2d_s;
  auto zten_3d_s2 = zten_3d_s;
  zten_res = zten_1d_s;
  zten_1d_s.Random(qn0);
  zten_1d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res,
      zrand()
  );
  zten_1d_s.Random(qn0);
  zten_1d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res,
      zrand()
  );
  zten_1d_s.Random(qnm1);
  zten_1d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2},
      &zten_res,
      zrand()
  );

  zten_res = zten_2d_s;
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res,
      zrand()
  );
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res,
      zrand()
  );
  zten_2d_s.Random(qnm1);
  zten_2d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2},
      &zten_res,
      zrand()
  );

  zten_res = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res,
      zrand()
  );
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res,
      zrand()
  );
  zten_3d_s.Random(qnm1);
  zten_3d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res
  );
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2},
      &zten_res,
      zrand()
  );
}
