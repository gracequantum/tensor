#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 21:16
#  
#  Description: GraceQ/tensor project. CMake file to control unittest.
# 
macro(add_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
  add_executable(${TEST_NAME}
      ${TEST_SRC})
  target_include_directories(${TEST_NAME} PUBLIC
      "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/src"
      "/usr/local/include"
      ${CFLAGS})
  target_link_libraries(${TEST_NAME}
      "-L/usr/local/lib"
      gtest gtest_main
      "${LINK_LIBS}" "${LINK_LIB_FLAGS}")
  add_test(NAME ${TEST_NAME} COMMAND "${TEST_NAME}" "${INPUT_ARGS}")
  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


# Test quantum number object.
add_unittest(test_qn test_qn.cc "" gqten "" "")

# Test quantum number sector object.
add_unittest(test_qnsector test_qnsector.cc "" gqten "" "")

# Test quantum number sector set object.
add_unittest(test_qnsectorset test_qnsectorset.cc "" gqten "" "")

# Test index object.
add_unittest(test_index test_index.cc "" gqten "" "")

# Test dense block labeled by the quantum numbers, QNBlock.
add_unittest(test_qnblock test_qnblock.cc "" gqten "" "")

# Test tensor with U1 symmetry, GQTensor.
add_unittest(test_gqtensor test_gqtensor.cc "" gqten "" "")

# Test tensor numerical functions.
add_unittest(test_numer_func
    test_numer_func.cc "${MATH_LIB_COMPILE_FLAGS}"
    gqten "${MATH_LIB_LINK_FLAGS}"
    "")
