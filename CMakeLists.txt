cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("bench" LANGUAGES CXX CUDA)
message(STATUS "C++ Compiler is: " ${CMAKE_CXX_COMPILER_ID})

set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} "~/hyacinth/")
add_compile_options("-fexceptions" "-Wall" "-Wextra" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "$<$<CONFIG:RELEASE>:-O3>")
link_directories("~/hyacinth/build")
link_libraries(OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK Eigen3::Eigen CUDA::cudart CUDA::cublas CUDA::cusolver libhyacinth.so)

file(GLOB sources "*.cpp" "*.cu")
enable_testing()

foreach (test ${sources})
  get_filename_component (src ${test} NAME_WE)
  message(STATUS "Source: " ${src})

  add_executable(${src}.app ${test})
  add_test(NAME ${src}.app COMMAND ${src}.app)
endforeach ()

