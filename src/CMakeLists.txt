# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 22:54
#  
#  Description: GraceQ/tensor project. CMake file to control the library source
#               code.
# 


add_library(gqten
  gqten.cc)

target_include_directories(gqten PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
