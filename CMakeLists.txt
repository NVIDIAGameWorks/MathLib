cmake_minimum_required(VERSION 3.16)

include(FetchContent)

file(GLOB MATHLIB_H
    "ml.h"
    "ml.hlsli"
)
source_group("MathLib" FILES ${MATHLIB_H})

file(GLOB MATHLIB_GUTS "Guts/*")
source_group("MathLib/Guts" FILES ${MATHLIB_GUTS})

set_property(SOURCE "ml.hlsli" PROPERTY VS_SETTINGS "ExcludedFromBuild=true")

if(WIN32)
    add_library(MathLib INTERFACE ${MATHLIB_H} ${MATHLIB_GUTS})
    set_property(TARGET MathLib PROPERTY FOLDER "MathLib")
else()
    add_library(MathLib INTERFACE)
endif()

target_include_directories(MathLib INTERFACE .)

# Download sse2neon on ARM
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "^arm")
    FetchContent_Declare(
        sse2neon
        GIT_REPOSITORY https://github.com/DLTcollab/sse2neon.git
        GIT_TAG master
        GIT_SHALLOW 1
    )

    message("MathLib: Downloading sse2neon...")
    FetchContent_MakeAvailable(sse2neon)

    target_include_directories(MathLib INTERFACE ${sse2neon_SOURCE_DIR})
endif()
