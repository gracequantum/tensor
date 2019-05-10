# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 22:54
#  
#  Description: GraceQ/tensor project. CMake file to control the library source
#               code.
# 


add_library(gqten
  ten_numer_func.cc
  gqten.cc)

target_include_directories(gqten
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PUBLIC "${MATH_LIB_COMPILE_FLAGS}")

target_link_libraries(gqten
  "${MATH_LIB_LINK_FLAGS}")
