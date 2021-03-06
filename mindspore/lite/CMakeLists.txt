cmake_minimum_required(VERSION 3.14)
project (Lite)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC vesion ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif ()

option(MS_VERSION_MAJOR "major version" 0)
option(MS_VERSION_MINOR "minor version" 7)
option(MS_VERSION_REVISION "revision version" 0)
option(CMAKE_BUILD_TYPE "build type" Release)
option(SUPPORT_TRAIN "if build for on-device train" off)
option(PLATFORM_ARM64 "if build device for arm64" off)
option(PLATFORM_ARM32 "if build device for arm32" off)
option(ENABLE_CONVERTER "if build converter" on)
option(ENABLE_FP16 "if build fp16 ops" off)
option(ENABLE_TOOLS "if build tools" on)
option(BUILD_TESTCASES "if build testcase" on)
option(SUPPORT_GPU "if support gpu" off)
option(SUPPORT_NPU "if support npu" off)
option(OFFLINE_COMPILE "if offline compile OpenCL kernel" off)
option(BUILD_MINDDATA_EXAMPLE "" on)
option(ENABLE_VERBOSE "" off)
option(ENABLE_SSE "if x86_64 support SSE instruction set" off)
option(ENABLE_AVX "if x86_64 support SSE instruction set" off)

set(DIR_PREFIX mindspore-lite)
set(MS_VERSION ${MS_VERSION_MAJOR}.${MS_VERSION_MINOR}.${MS_VERSION_REVISION})
set(MAIN_DIR ${DIR_PREFIX}-${MS_VERSION})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_VERSION_MAJOR=${MS_VERSION_MAJOR} -DMS_VERSION_MINOR=${MS_VERSION_MINOR} -DMS_VERSION_REVISION=${MS_VERSION_REVISION}")
set(BUILD_MINDDATA "lite_cv" CACHE STRING "off, lite, lite_cv, wrapper or full")
set(BUILD_LITE "on")
set(PLATFORM_ARM "off")
if (PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(PLATFORM_ARM "on")
    #set for cross-compiling toolchain
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

if (SUPPORT_GPU)
    set(PROCESS_UNIT gpu)
elseif (SUPPORT_NPU)
    set(PROCESS_UNIT npu)
else ()
    set(PROCESS_UNIT cpu)
endif ()

if (SUPPORT_NPU)
    set(DDK_PATH "$ENV{HWHIAI_DDK}/ddk/ai_ddk_lib")
    if (PLATFORM_ARM64)
        set(DDK_LIB_PATH ${DDK_PATH}/lib64)
    elseif (PLATFORM_ARM32)
        set(DDK_LIB_PATH ${DDK_PATH}/lib)
    endif()
    add_compile_definitions(SUPPORT_NPU)
endif()

add_compile_definitions(NO_DLIB)
add_compile_options(-fPIC)

if (SUPPORT_TRAIN)
    if (PLATFORM_ARM64)
        set(RUNTIME_COMPONENT_NAME train-android-aarch64)
    elseif (PLATFORM_ARM32)
        set(RUNTIME_COMPONENT_NAME train-android-aarch32)
    elseif (WIN32)
        set(RUNTIME_COMPONENT_NAME train-win-x64)
        set(CONVERTER_COMPONENT_NAME train-converter-win-x64)
    else ()
        set(RUNTIME_COMPONENT_NAME train-linux-x64)
        set(CONVERTER_COMPONENT_NAME train-converter-linux-x64)
    endif ()
else ()
    if (PLATFORM_ARM64)
        set(RUNTIME_COMPONENT_NAME inference-android-aarch64)
    elseif (PLATFORM_ARM32)
        set(RUNTIME_COMPONENT_NAME inference-android-aarch32)
    elseif (WIN32)
        if ("${X86_64_SIMD}" STREQUAL "off")
            set(RUNTIME_COMPONENT_NAME inference-win-x64)
        else ()
            set(RUNTIME_COMPONENT_NAME inference-win-x64-${X86_64_SIMD})
        endif()
        set(CONVERTER_COMPONENT_NAME converter-win-x64)
    else ()
        if ("${X86_64_SIMD}" STREQUAL "off")
            set(RUNTIME_COMPONENT_NAME inference-linux-x64)
        else ()
            set(RUNTIME_COMPONENT_NAME inference-linux-x64-${X86_64_SIMD})
        endif()
        set(CONVERTER_COMPONENT_NAME converter-linux-x64)
    endif()
endif ()

string(REPLACE "/mindspore/lite" "" TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CORE_DIR ${TOP_DIR}/mindspore/core)
set(CCSRC_DIR ${TOP_DIR}/mindspore/ccsrc)
include_directories(${TOP_DIR})
include_directories(${CORE_DIR})
include_directories(${CORE_DIR}/ir)
include_directories(${CCSRC_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/kernel/arm)
include_directories(${TOP_DIR}/third_party)
include_directories(${CMAKE_BINARY_DIR})

include(${TOP_DIR}/cmake/utils.cmake)
include(${TOP_DIR}/cmake/dependency_utils.cmake)
include(${TOP_DIR}/cmake/dependency_securec.cmake)
include(${TOP_DIR}/cmake/external_libs/flatbuffers.cmake)
if (SUPPORT_GPU)
    include(${TOP_DIR}/cmake/external_libs/opencl.cmake)
endif()

if (ENABLE_CONVERTER OR BUILD_MINDDATA STREQUAL "full")
    include(${TOP_DIR}/cmake/external_libs/json.cmake)
endif()

file(GLOB FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/schema/*.fbs)
ms_build_flatbuffers_lite(FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/schema/ fbs_src ${CMAKE_BINARY_DIR}/schema "")
ms_build_flatbuffers_lite(FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/schema/ fbs_inner_src ${CMAKE_BINARY_DIR}/schema/inner "inner")

string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else ()
    ## enable for binscope for release
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    if (NOT WIN32)
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,-z,noexecstack ${CMAKE_SHARED_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,relro,-z,now -Wl,-z,noexecstack ${CMAKE_EXE_LINKER_FLAGS}")
    endif ()
endif ()
if (ENABLE_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE on)
endif ()
if (SUPPORT_TRAIN)
    add_compile_definitions(SUPPORT_TRAIN)
endif ()
if (ENABLE_NEON)
    add_compile_definitions(ENABLE_NEON)
endif ()
if (ENABLE_FP16)
    add_compile_definitions(ENABLE_FP16)
endif ()
if (SUPPORT_GPU)
    gene_opencl(${CMAKE_CURRENT_SOURCE_DIR})
    add_definitions(-DUSE_OPENCL_WRAPPER)
    add_definitions(-DMS_OPENCL_PROFILE=false)
    add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=200)
    add_compile_definitions(SUPPORT_GPU)
    if (OFFLINE_COMPILE)
        add_compile_definitions(PROGRAM_WITH_IL)
    endif ()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/opencl-headers-src/)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/opencl-clhpp-src/include)
endif ()

if (WIN32)
    add_compile_definitions(LITE_EXPORTS)
    add_compile_definitions(BUILDING_DLL)
endif ()

if (ENABLE_CONVERTER)
    if (PLATFORM_ARM)
        MESSAGE(FATAL_ERROR "Cannot build converter in arm platform")
    endif ()
    include_directories(${PYTHON_INCLUDE_DIRS})
    include(${TOP_DIR}/cmake/external_libs/eigen.cmake)
    include(${TOP_DIR}/cmake/external_libs/protobuf.cmake)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/converter)
endif ()

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    if (NOT DEFINED ENV{ANDROID_NDK})
        message(FATAL_ERROR "env ANDROID_NDK should be setted for ARM compile")
    endif ()
    add_compile_definitions(ENABLE_ARM)
endif ()
if (PLATFORM_ARM32)
    add_definitions(-mfloat-abi=softfp -mfpu=neon)
    add_compile_definitions(ENABLE_ARM32)
endif ()
if (PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
    if (ENABLE_FP16)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16")
    endif ()
endif ()

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
  if (ENABLE_CONVERTER)
    set(BUILD_MINDDATA "off")
  endif()
endif()

if (NOT PLATFORM_ARM32 AND NOT PLATFORM_ARM64)
    if ("${X86_64_SIMD}" STREQUAL "sse")
        add_compile_definitions(ENABLE_SSE)
    endif ()
    if ("${X86_64_SIMD}" STREQUAL "avx")
        add_compile_definitions(ENABLE_SSE)
        add_compile_definitions(ENABLE_AVX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mfma")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")
    endif ()
endif ()

if (BUILD_MINDDATA STREQUAL "lite" OR BUILD_MINDDATA STREQUAL "full" OR BUILD_MINDDATA STREQUAL "wrapper")
    add_compile_definitions(ENABLE_ANDROID)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/minddata)
endif ()

if (BUILD_MINDDATA STREQUAL "lite_cv")
    add_compile_definitions(ENABLE_ANDROID)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/minddata)
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nnacl)
if (ENABLE_TOOLS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark)
    if (SUPPORT_TRAIN)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_train)
    endif ()
endif ()
if (NOT WIN32)
    if (ENABLE_TOOLS)
        if (NOT PLATFORM_ARM32 AND NOT PLATFORM_ARM64)
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/schema_gen)
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/cropper)
        endif ()
    endif ()
    if (BUILD_TESTCASES)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
    endif ()
endif ()

include(${TOP_DIR}/cmake/package_lite.cmake)

