cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("examples" LANGUAGES CXX)
option (USE_NCCL "Use NCCL" ON)
option (BOOTSTRAP_POSIX "Use POSIX fork BOOTSTRAP" ON)
option (BOOTSTRAP_MPI "Use MPI BOOTSTRAP" ON)

set(Hyacin_DIR "/home/ma/hyacin-build/lib/cmake")
find_package(Hyacin REQUIRED)
find_package(BLAS REQUIRED)

if(USE_NCCL)
  if(BOOTSTRAP_MPI)
    find_package(MPI REQUIRED)
  endif()
  set(cpp_sources zgeqrf_cusolver.cpp zgevd_vendor.cpp xlra_example.cpp xsvd_example.cpp xsvd_1drow_example.cpp xsvd_2d_example.cpp bootstrap_check.cpp)
else()
  set(cpp_sources zgeqrf_cusolver.cpp xlra_example.cpp xsvd_example.cpp)
endif()

set_source_files_properties(${cpp_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-fexceptions")

foreach (exe ${cpp_sources})
  get_filename_component(src ${exe} NAME_WE)
  add_executable(${src}.app ${exe})
  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${src}.app PRIVATE Hyacin::hyacin BLAS::BLAS)

  if(USE_NCCL)
    if(NOT BOOTSTRAP_POSIX)
      target_compile_definitions(${src}.app PRIVATE "BOOTSTRAP_NO_POSIX")
    endif()
    
    if(BOOTSTRAP_MPI)
      target_link_libraries(${src}.app PRIVATE MPI::MPI_CXX)
    else()
      target_compile_definitions(${src}.app PRIVATE "BOOTSTRAP_NO_MPI")
    endif()
  else()
    target_compile_definitions(${src}.app PRIVATE "NO_NCCL")
  endif()
endforeach ()

