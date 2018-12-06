cmake_minimum_required(VERSION 3.12)
project(mpi_lab_2 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_COMPILER mpicc)

link_libraries(m)

add_executable(mpi_lab_2 main.c)