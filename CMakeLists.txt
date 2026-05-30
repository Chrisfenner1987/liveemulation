# FilmGrain (Stochastic) OFX - cross-platform build.
#   macOS        -> Metal GPU path        (Contents/MacOS)
#   Windows/Linux-> CUDA GPU path (NVIDIA) (Contents/Win64 / Contents/Linux-x86-64)
#
# The macOS reference build is the Makefile; this CMake file is the portable build
# used to produce the Windows (and Linux) binaries. See WINDOWS_BUILD.md.
cmake_minimum_required(VERSION 3.20)
project(FilmGrainOFX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OFX_ROOT   ${CMAKE_CURRENT_SOURCE_DIR}/openfx)
set(OFX_INC    ${OFX_ROOT}/include)
set(OFXS_INC   ${OFX_ROOT}/Support/include)
set(OFXS_PLUG  ${OFX_ROOT}/Support/Plugins/include)
set(OFXS_LIB   ${OFX_ROOT}/Support/Library)

file(GLOB SUPPORT_SOURCES ${OFXS_LIB}/*.cpp)

# ---- platform GPU sources / bundle layout ---------------------------------
if(APPLE)
    set(BUNDLE_ARCH_DIR "MacOS")
    set(GPU_SOURCES src/FilmGrainMetal.mm)
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    endif()
else()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Explicit list (the "all-major" keyword is NOT expanded by the VS generator).
        # 52=Maxwell 61=Pascal(10xx) 75=Turing(16/20xx) 86=Ampere(30xx) 89=Ada(40xx).
        set(CMAKE_CUDA_ARCHITECTURES "52;61;75;86;89")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GPU_SOURCES src/FilmGrainCuda.cu)
    if(WIN32)
        set(BUNDLE_ARCH_DIR "Win64")
    else()
        set(BUNDLE_ARCH_DIR "Linux-x86-64")
    endif()
endif()

add_library(FilmGrain MODULE
    src/FilmGrain.cpp
    ${GPU_SOURCES}
    ${SUPPORT_SOURCES}
)

target_include_directories(FilmGrain PRIVATE ${OFX_INC} ${OFXS_INC} ${OFXS_PLUG} ${CMAKE_CURRENT_SOURCE_DIR}/src)

# OFX_SUPPORTS_OPENGLRENDER gates all GPU (Metal/CUDA) plumbing in the Support lib
# and must be defined for every translation unit (it changes struct layouts).
target_compile_definitions(FilmGrain PRIVATE OFX_SUPPORTS_OPENGLRENDER)

set_target_properties(FilmGrain PROPERTIES
    PREFIX ""
    OUTPUT_NAME "FilmGrain"
    SUFFIX ".ofx"
)

if(APPLE)
    target_link_libraries(FilmGrain PRIVATE "-framework Metal" "-framework Foundation" "-framework AppKit")
else()
    target_compile_definitions(FilmGrain PRIVATE FILMGRAIN_CUDA)
    # Single .cu, no cross-TU device calls -> separable compilation not needed
    # (and its extra device-link step can fail on some toolchains).
    set_target_properties(FilmGrain PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_link_libraries(FilmGrain PRIVATE CUDA::cudart)
endif()

# ---- assemble the .ofx.bundle ---------------------------------------------
set(BUNDLE_DIR ${CMAKE_BINARY_DIR}/FilmGrain.ofx.bundle)
add_custom_command(TARGET FilmGrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_DIR}/Contents/${BUNDLE_ARCH_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FilmGrain> ${BUNDLE_DIR}/Contents/${BUNDLE_ARCH_DIR}/FilmGrain.ofx
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist ${BUNDLE_DIR}/Contents/Info.plist
    COMMENT "Assembling FilmGrain.ofx.bundle (${BUNDLE_ARCH_DIR})"
)
