cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("bench" LANGUAGES CXX)
message(STATUS "C++ Compiler is: " ${CMAKE_CXX_COMPILER_ID})

find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${MAGMA_INCLUDE_DIRS})
add_compile_options("-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "$<$<CONFIG:RELEASE>:-O3>")
link_libraries(OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK lapacke ${MAGMA_LINK_LIBRARIES})

file(GLOB sources "*.cpp")
enable_testing()

foreach (test ${sources})
    get_filename_component (src ${test} NAME_WE)
    message(STATUS "Source: " ${src})

    add_executable(${src} ${test})
    add_test(NAME ${src} COMMAND ${src})
endforeach ()



