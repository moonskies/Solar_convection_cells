cmake_minimum_required(VERSION 3.20)
project(Solar_convection_cells3)

set(CMAKE_CXX_STANDARD 17)

# 1. Find the OpenMP package
find_package(OpenMP REQUIRED)

# 2. Add your executable
add_executable(Solar_convection_cells3 main.cpp)

# 3. Link OpenMP
if(OpenMP_CXX_FOUND)
    # This connects the library to your target
    target_link_libraries(Solar_convection_cells3 PUBLIC OpenMP::OpenMP_CXX)

    # CRITICAL FIX FOR MINGW:
    # We explicitly force the compiler and linker to use the -fopenmp flag.
    # Without this, MinGW sometimes drops the flag during the link step.
    target_compile_options(Solar_convection_cells3 PUBLIC -fopenmp -O3 -march=native)
    target_link_options(Solar_convection_cells3 PUBLIC -fopenmp)
endif()