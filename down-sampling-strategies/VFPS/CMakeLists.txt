cmake_minimum_required(VERSION 2.6)
project(Voxel_filter)
 
find_package(PCL 1.12 REQUIRED)
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(Voxel_filter Voxel_filter.cpp)
 
target_link_libraries (Voxel_filter ${PCL_LIBRARIES})
 
install(TARGETS Voxel_filter RUNTIME DESTINATION bin)
