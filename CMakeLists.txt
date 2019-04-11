cmake_minimum_required(VERSION 2.8.3)
project(elised_plus_plus)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -std=c++11 -O3 -march=native")

# search for everything we need to build the package
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cmake_modules
  tf
)

# find_package makes the ${..._INCLUDE_DIRS} ${..._LIBRARIES} variables we use later
set(OpenCV_DIR "usr/local/opencv2/share/OpenCV")
find_package(OpenCV REQUIRED)

find_package(Eigen REQUIRED)

# export the dependencis of this package for who ever depends on us
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
  DEPENDS
        OpenCV
        Eigen
  LIBRARIES
        ${PROJECT_NAME}
)

file(GLOB ${PROJECT_NAME}_CODE_BASE
  "src/code_base/*.cpp"
  "src/code_base/*.cc"
)

# tell catkin where to find the headers for this project
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

# add_library(${PROJECT_NAME} STATIC
#   ${${PROJECT_NAME}_CODE_BASE}
# )
add_executable(${PROJECT_NAME} src/test.cpp ${${PROJECT_NAME}_CODE_BASE})  

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
)


