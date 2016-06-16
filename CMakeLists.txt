cmake_minimum_required (VERSION 2.6)
project(saliency)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package( OpenCV REQUIRED )
find_package( TBB )
#find_package(OpenMP REQUIRED)

#add_definitions(${OpenMP_CXX_FLAGS})
add_definitions( -g -std=c++0x )

add_subdirectory (src)
