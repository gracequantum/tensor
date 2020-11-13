#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 21:16
#  
#  Description: GraceQ/tensor project. CMake file to control unittest.
# 
if(NOT GQTEN_USE_EXTERNAL_HPTT_LIB)
  set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/hptt/libhptt.a")
endif()


macro(add_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS)
  add_executable(${TEST_NAME}
    ${TEST_SRC})

    target_include_directories(${TEST_NAME}
      PRIVATE ${GQTEN_HEADER_PATH}
      PRIVATE ${GQTEN_SRC_PATH}
      PRIVATE ${CFLAGS})
    target_link_libraries(${TEST_NAME}
      GTest::GTest GTest::Main
      ${hptt_LIBRARY}
      "${LINK_LIBS}" "${LINK_LIB_FLAGS}")

    add_test(
      NAME ${TEST_NAME}
      COMMAND "${TEST_NAME}")

  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


## Test quantum number related objects
# Test quantum number vaules
add_unittest(test_qnval_u1 "test_gqtensor/test_qnval/test_qnval_u1.cc" "" "" "")
# Test quantum number
add_unittest(test_qn "test_gqtensor/test_qn.cc" "" "" "")

## Test quantum number sector object.
add_unittest(test_qnsct "test_gqtensor/test_qnsct.cc" "" "" "")

## Test quantum number sector set object.
#add_unittest(test_qnsectorset test_qnsectorset.cc "" "" "${MATH_LIB_LINK_FLAGS}")

## Test index object.
add_unittest(test_index "test_gqtensor/test_index.cc" "" "" "")

## Test dense block labeled by the quantum numbers, QNBlock.
#add_unittest(test_qnblock test_qnblock.cc "" "" "${MATH_LIB_LINK_FLAGS}")

## Test tensor with U1 symmetry, GQTensor.
#add_unittest(test_gqtensor test_gqtensor.cc "" "" "${MATH_LIB_LINK_FLAGS}")

## Test tensor numerical functions.
## Test tensor contraction.
#add_unittest(test_ten_ctrct
  #test_ten_ctrct.cc "${MATH_LIB_COMPILE_FLAGS}"
  #"" "${MATH_LIB_LINK_FLAGS}")
## Test tensor linear combination.
#add_unittest(test_ten_lincmb
  #test_ten_lincmb.cc "${MATH_LIB_COMPILE_FLAGS}"
  #"" "${MATH_LIB_LINK_FLAGS}")

## Test tensor SVD.
#add_unittest(test_ten_svd
  #test_ten_svd.cc "${MATH_LIB_COMPILE_FLAGS}"
  #"" "${MATH_LIB_LINK_FLAGS}")

## Test tensor expansion.
#add_unittest(test_ten_expand
  #test_ten_expand.cc "${MATH_LIB_COMPILE_FLAGS}"
  #"" "${MATH_LIB_LINK_FLAGS}")

## Test index combination.
#add_unittest(test_index_combine
  #test_index_combine.cc "${MATH_LIB_COMPILE_FLAGS}"
  #"" "${MATH_LIB_LINK_FLAGS}")
