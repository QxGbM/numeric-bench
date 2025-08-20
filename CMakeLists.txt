cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("bench" LANGUAGES CXX)
message(STATUS "C++ Compiler is: " ${CMAKE_CXX_COMPILER_ID})

set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(MAGMA_INCLUDE_DIRS "/home/ma/magma-2.9.0/include")
set(MAGMA_LIB_PATHS "/home/ma/magma-2.9.0/lib")
set(MAGMA_LINK_LIBRARIES "-lmagma -lmagma_sparse")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${MAGMA_INCLUDE_DIRS})
add_compile_options("-fexceptions" "-Wall" "-Wextra" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "$<$<CONFIG:RELEASE>:-O3>")
link_directories(${MAGMA_LIB_PATHS})
link_libraries(OpenMP::OpenMP_CXX MKL::MKL CUDA::cudart CUDA::cublas CUDA::cusolver ${MAGMA_LINK_LIBRARIES})
#link_libraries(OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK CUDA::cudart CUDA::cublas CUDA::cusolver lapacke ${MAGMA_LINK_LIBRARIES})

file(GLOB sources "*.cpp")
enable_testing()

foreach (test ${sources})
  get_filename_component (src ${test} NAME_WE)
  message(STATUS "Source: " ${src})

  add_executable(${src}.app ${test})
  add_test(NAME ${src}.app COMMAND ${src}.app)
endforeach ()

