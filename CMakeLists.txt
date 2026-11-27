cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project("examples" LANGUAGES CXX)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  message(STATUS "CMAKE_CUDA_HOST_COMPILER not specified; defaulting to ${CMAKE_CUDA_HOST_COMPILER}")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90a;100a;103a;120a;121a")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not specified; defaulting to ${CMAKE_CUDA_ARCHITECTURES}")
endif()
enable_language(CUDA)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:~/cmake_install/lib/pkgconfig")
find_package(PkgConfig REQUIRED)
find_package(Hyacin REQUIRED PATHS "~/cmake_install/lib/cmake")
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
pkg_check_modules(MAGMA REQUIRED magma)

set(magma_PREFIX ~/cmake_install)
find_package(kblasgpu REQUIRED CONFIG PATHS "~/cmake_install" PATH_SUFFIXES "lib/cmake/kblasgpu" "lib64/cmake/kblasgpu")

set(cpp_sources
  syrk_bench.cpp xgeqp3_magma.cpp xgesvd_cusolver.cpp xgesvdr_cusolver.cpp xgesvj_magma_batch.cpp xgersvd_kblas_batch.cpp
  xlra_hyacin.cpp xsvd_hyacin.cpp xgevd_vendor.cpp xsvd_1drow_hyacin.cpp bootstrap_check.cpp
  tests/vendor_checks.cpp tests/hyacin_checks.cpp)
set(cu_sources
  blr_cusolver.cu hss_cusolver.cu hss_rsvd.cu hss_hyacin.cu
  tests/hss_rsvd_checks.cu tests/hss_hyacin_checks.cu)

enable_testing()
add_test(NAME vendor_host_checks COMMAND vendor_checks.app)
add_test(NAME hyacin_host_checks COMMAND hyacin_checks.app)
add_test(NAME hyacin_gpu_checks COMMAND hyacin_checks.app --gpu)
add_test(NAME hss_rsvd_gpu_checks COMMAND hss_rsvd_checks.app)
add_test(NAME hss_hyacin_gpu_checks COMMAND hss_hyacin_checks.app)
set_tests_properties(
  vendor_host_checks hyacin_host_checks hyacin_gpu_checks hss_rsvd_gpu_checks hss_hyacin_gpu_checks
  PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 120)

set_source_files_properties(${cpp_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-Wfatal-errors;-fexceptions")
set_source_files_properties(${cu_sources} PROPERTIES COMPILE_OPTIONS "-Werror=all-warnings;-Xcompiler=-fexceptions")
set(all_sources ${cpp_sources} ${cu_sources})

foreach (exe ${all_sources})
  get_filename_component(src ${exe} NAME_WE)
  add_executable(${src}.app ${exe})
  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${MAGMA_INCLUDE_DIRS})
  target_link_directories(${src}.app PRIVATE ${MAGMA_LIBRARY_DIRS})
  target_link_libraries(${src}.app PRIVATE MPI::MPI_CXX Hyacin::hyacin BLAS::BLAS KBLASGPU::kblasgpu)
endforeach ()
