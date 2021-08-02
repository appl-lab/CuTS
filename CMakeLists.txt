cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(graph_match_gpu LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
find_package(OpenMP)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_executable(cuts inc/common.h inc/device_funcs.h inc/free_memories.h
        inc/gpu_memory_allocation.h inc/graph.h inc/host_funcs.h inc/score.h inc/util.h
        src/common.cpp src/device_funcs.cu src/free_memories.cu src/gpu_memory_allocation.cu src/graph.cpp
        src/host_funcs.cu src/score.cpp src/util.cpp main.cu)
target_compile_options(cuts PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
#target_compile_options(cuts PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)
add_executable(cuts_mpi inc/common.h inc/device_funcs.h inc/free_memories.h
        inc/gpu_memory_allocation.h inc/graph.h inc/score.h inc/util.h
        src/common.cpp src/device_funcs.cu src/free_memories.cu src/gpu_memory_allocation.cu src/graph.cpp
        src/score.cpp src/util.cpp main_mpi.cu)
target_compile_options(cuts_mpi PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)
#target_compile_options(cuts_mpi PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)
target_link_libraries(cuts_mpi ${MPI_LIBRARIES})
