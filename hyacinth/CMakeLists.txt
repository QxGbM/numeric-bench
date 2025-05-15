cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("hyacinth" LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(hyacinth SHARED cpotrfp_gpu.cu)
add_library(hyacinth_static STATIC cpotrfp_gpu.cu)
set_target_properties(hyacinth hyacinth_static PROPERTIES CUDA_ARCHITECTURES all-major)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options("-Xcompiler=-fPIC" "-Xcompiler=-Wall" "-Xcompiler=-Wextra" "-Xcompiler=-fexceptions")
link_libraries(CUDA::cudart CUDA::cublas)
