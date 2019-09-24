#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 22:54
#  
#  Description: GraceQ/tensor project. CMake file to control the library source code.
# 
if(NOT GQTEN_USE_EXTERNAL_HPTT_LIB)
  set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/hptt/include")
endif()


set(GQTEN_HEADERS_DIR ${GQTEN_HEADER_PATH}/gqten)
set(GQTEN_BASIC_SRC
  utils.cc
  timer.cc
  ten_trans.cc
  qn.cc
  qnsector.cc
  index.cc)


  add_library(gqten
    ${GQTEN_BASIC_SRC})

  target_include_directories(gqten
    PRIVATE ${GQTEN_HEADER_PATH}
    PRIVATE ${hptt_INCLUDE_DIR})

  target_compile_options(gqten
    PRIVATE ${MATH_LIB_COMPILE_FLAGS})


install(
  DIRECTORY ${GQTEN_HEADERS_DIR}
  DESTINATION ${GQTEN_INCLUDE_DEST})
install(
  TARGETS gqten
  DESTINATION "${GQTEN_LIB_DEST}")
