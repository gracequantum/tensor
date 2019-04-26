# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 21:16
#  
#  Description: GraceQ/tensor project. CMake file to control unittest.
# 
macro(add_unittest
    TEST_NAME TEST_SRC LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
  add_executable(${TEST_NAME}
      ${TEST_SRC})
  target_include_directories(${TEST_NAME} PUBLIC
      "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(${TEST_NAME}
      gtest gtest_main
      "${LINK_LIBS}" "${LINK_LIB_FLAGS}")
  add_test(NAME ${TEST_NAME} COMMAND "${TEST_NAME}" "${INPUT_ARGS}")
  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


# Test quantum number object.
add_unittest(test_qn test_qn.cc gqten "" "")

# Test quantum number sector object.
add_unittest(test_qnsector test_qnsector.cc gqten "" "")
