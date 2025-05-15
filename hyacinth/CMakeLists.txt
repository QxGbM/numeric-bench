cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("hyacinth" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CPP_SOURCES align_ld.cpp)
set(CU_SOURCES cpotrfp_gpu.cu complex_fp_i8.cu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_libraries(CUDA::cudart CUDA::cublas)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Wall;-Wextra;-fexceptions")
set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fexceptions")

add_library(hyacinth SHARED ${CPP_SOURCES} ${CU_SOURCES})
set_target_properties(hyacinth PROPERTIES CUDA_ARCHITECTURES all-major)

