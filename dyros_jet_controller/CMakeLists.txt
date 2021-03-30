cmake_minimum_required(VERSION 2.8.3)
project(dyros_jet_controller)

# set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

#add_compile_options(-ofast)

find_package(catkin REQUIRED COMPONENTS
dyros_jet_msgs
imu_3dm_gx4
rt_dynamixel_msgs
geometry_msgs
roscpp
roslib
sensor_msgs
std_msgs
realtime_tools
eigen_conversions
actionlib
)
find_package(Boost REQUIRED COMPONENTS system thread)

#conan pkg
find_package(rbdl_jet REQUIRED)
include_directories(${rbdl_jet_INCLUDE_DIRS})

find_package(qpOASES REQUIRED)
include_directories(${qpOASES_INCLUDE_DIRS})

find_package(toml11 REQUIRED)
include_directories(${toml11_INCLUDE_DIRS})

find_package(h5cpp11 REQUIRED)
include_directories(${h5cpp11_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

#find_package(boost REQUIRED)
catkin_package(
    DEPENDS   dyros_jet_msgs imu_3dm_gx4 rt_dynamixel_msgs geometry_msgs roscpp sensor_msgs std_msgs realtime_tools eigen_conversions
    CATKIN_DEPENDS # TODO
    INCLUDE_DIRS include
    LIBRARIES # TODO
    )
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    )


file(GLOB_RECURSE HeaderFiles "include/*.h")
add_custom_target(dyros_jet_control_headers SOURCES ${HeaderFiles})

# file(GLOB CVXGEN_SRC
#     "include/cvxgen_6_8_0/cvxgen/*.h"
#     "include/cvxgen_6_8_0/cvxgen/solver.c"
#     "include/cvxgen_6_8_0/cvxgen/matrix_support.c"
#     "include/cvxgen_6_8_0/cvxgen/ldl.c"
#     "include/cvxgen_6_8_0/cvxgen/util.c"
#     )

file(GLOB BRIDGE_SRC
    "src/real_robot_interface.cpp"
    "src/simulation_interface.cpp" 
    "src/mujoco_interface.cpp"
    "src/control_base.cpp"
    #"include/dyros_jet_controller/dyros_jet_model.h"
    #  "src/hardware_interface.cpp"
    )
file(GLOB WALKING_SRC
    "src/walking_controller_hw.cpp"
    "src/state_estimator.cpp"
    )
file(GLOB CONTROLLER_SRC
    "src/dyros_jet_model.cpp"
    "src/walking_controller_hw.cpp"
    "src/task_controller.cpp"
    "src/haptic_controller.cpp"
    "src/joint_controller.cpp"
    "src/moveit_controller.cpp"
    "src/quadraticprogram.cpp"
    "src/retarget_controller.cpp"
    )

add_library(dyros_jet_controller_lib SHARED
    ${BRIDGE_SRC}
    ${CONTROLLER_SRC}
    ${WALKING_SRC}
    # ${CVXGEN_SRC}
    #  ${UPPER_SRC}
    )

target_link_libraries(dyros_jet_controller_lib
        ${catkin_LIBRARIES}
        ${Boost_LIBRARIES}
        ${qpOASES_LIBRARIES}
        ${rbdl_jet_LIBRARIES}
        ${toml11_LIBRARIES}
        ${h5cpp11_LIBRARIES}
        ${Eigen3_LIBRARIES}
        )

# controller main
add_executable(dyros_jet_controller src/dyros_jet_controller.cpp)
target_link_libraries(dyros_jet_controller
    dyros_jet_controller_lib
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    ${h5cpp11_LIBRARIES}
    # ${Eigen3_LIBRARIES}
    # ${CVXGEN_SRC}
    )

add_dependencies(dyros_jet_controller_lib ${catkin_EXPORTED_TARGETS})
add_dependencies(dyros_jet_controller ${catkin_EXPORTED_TARGETS})


SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)
