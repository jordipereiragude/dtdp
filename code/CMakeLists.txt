cmake_minimum_required(VERSION 3.5)

cmake_policy(SET CMP0048 NEW)
project(stochasticALBP VERSION 1.0)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 8.0.1)
FetchContent_MakeAvailable(fmtlib)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(CMakeBuildTypes)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE}) 

set(CMAKE_CXX_FLAGS "-DIL_STD -Wall -Wno-unknown-pragmas -fexceptions -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb -fno-omit-frame-pointer -fsanitize=address")

find_package(cplex)
find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options)

include_directories(SYSTEM ${CPLEX_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(AFTER .)
include_directories(SYSTEM ../3rd/fmt/include)
link_directories(${Boost_LIBRARY_DIRS})

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(EXTRA_LIBRARIES dl)
else()
  set(EXTRA_LIBRARIES rt stdc++fs dl)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(CMakeBuildTypes)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE}) 



add_executable(ip main.cpp instance.cpp options.cpp logging.cpp ip.cpp greedy.cpp solution.cpp random.cpp dw.cpp iterative_relaxation.cpp)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_libraries(ip ${CPLEX_LIBRARIES} ${Boost_LIBRARIES} dl fmt::fmt  -ld_classic)
  target_compile_definitions(ip PUBLIC "IS_MACOS")
else()
  target_link_libraries(ip ${CPLEX_LIBRARIES} ${Boost_LIBRARIES} rt stdc++fs dl fmt::fmt)
endif()
