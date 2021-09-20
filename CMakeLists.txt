cmake_minimum_required(VERSION 3.12)
project(alex)

set(CMAKE_CXX_STANDARD 14)
include(FetchContent)
include(ExternalProject)

if(MSVC)
    set(CMAKE_CXX_FLAGS "/O2 /arch:AVX2 /W1 /EHsc")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "-O3 -xHost")
else()
    # clang and gcc
    set(CMAKE_CXX_FLAGS "-lnuma -lrt -lpthread -ldl -mrtm -msse4.1 -mavx2 -O3 -g -march=native")
endif()

add_definitions(-DPMEM)
add_definitions(-DPMDK)
message(STATUS "Persistence support: PMDK")

##################### PMDK ####################
set(PMDK_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/pmdk")
ExternalProject_Add(PMDK
        GIT_REPOSITORY https://github.com/HaoPatrick/pmdk.git
        GIT_TAG addr-patch
	      BUILD_IN_SOURCE 1
        BUILD_COMMAND $(MAKE) EXTRA_CFLAGS=-Wno-error
        PREFIX ${PMDK_PREFIX}
        CONFIGURE_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        )

include_directories(${PMDK_PREFIX}/src/PMDK/src/include)
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  link_directories(${PMDK_PREFIX}/src/PMDK/src/debug)
else ()
  link_directories(${PMDK_PREFIX}/src/PMDK/src/nondebug)
endif ()

##################### Epoch Reclaim ####################
FetchContent_Declare(
        epoch_reclaimer
        GIT_REPOSITORY https://github.com/XiangpengHao/epoch-reclaimer.git
        GIT_TAG master
)
if (NOT epoch_reclaimer_POPULATED)
  FetchContent_Populate(epoch_reclaimer)
  include_directories(${epoch_reclaimer_SOURCE_DIR})
endif ()

include_directories(src/core)
list(APPEND libs_to_link pmemobj pmem tbb papi tcmalloc_minimal)

add_executable(benchmark src/benchmark/main.cpp)            
add_dependencies(benchmark PMDK)
target_link_libraries(benchmark PRIVATE ${libs_to_link})

#add_executable(partition src/benchmark/partition.cpp)
#add_dependencies(partition PMDK)
#target_link_libraries(partition PRIVATE ${libs_to_link})

#set(MAX_FREEZE_RETRY 1 CACHE STRING "MAX retry on frozen node")
#target_compile_definitions(lbtree-test PRIVATE MAX_FREEZE_RETRY=${MAX_FREEZE_RETRY})
#set(ENABLE_MERGE 0 CACHE STRING "MAX retry on frozen node")
#message(STATUS "ENABLE_MERGE: " ${ENABLE_MERGE})
#target_compile_definitions(lbtree-test PRIVATE ENABLE_MERGE=${ENABLE_MERGE})