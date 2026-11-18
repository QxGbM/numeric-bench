cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("examples" LANGUAGES CXX CUDA)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:~/cmake_install/lib/pkgconfig")

find_package(PkgConfig REQUIRED)
find_package(Hyacin REQUIRED PATHS "~/cmake_install/lib/cmake")
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)

set(cpp_sources syrk_bench.cpp xgeqp3_magma.cpp xgesvd_cusolver.cpp xgesvdr_cusolver.cpp xgevd_vendor.cpp xlra_hyacin.cpp xsvd_hyacin.cpp xsvd_1drow_hyacin.cpp bootstrap_check.cpp)
set(cu_sources hss_cusolver.cu)

include(CTest)
if(BUILD_TESTING)
  list(APPEND cpp_sources tests/vendor_checks.cpp)
  add_test(NAME vendor_host_checks COMMAND vendor_checks.app)
endif()

set_source_files_properties(${cpp_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-Wfatal-errors")
set_source_files_properties(${cu_sources} PROPERTIES COMPILE_OPTIONS "-Werror=all-warnings")
list(APPEND all_sources ${cpp_sources} ${cu_sources})

foreach (exe ${all_sources})
  get_filename_component(src ${exe} NAME_WE)
  add_executable(${src}.app ${exe})
  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${MAGMA_INCLUDE_DIRS})
  target_link_directories(${src}.app PRIVATE ${MAGMA_LIBRARY_DIRS})
  target_link_libraries(${src}.app PRIVATE MPI::MPI_CXX Hyacin::hyacin BLAS::BLAS magma_sparse magma)
endforeach ()

