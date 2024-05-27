cmake_minimum_required(VERSION 2.6)
project(RandomSample)
 
find_package(PCL 1.12 REQUIRED)
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(RandomSample RandomSample.cpp)
 
target_link_libraries (RandomSample ${PCL_LIBRARIES})
 
install(TARGETS RandomSample RUNTIME DESTINATION bin)
