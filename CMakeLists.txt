cmake_minimum_required(VERSION 3.15)
project(mario_dqn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 1: DQN on CartPole. Dependency-free (self-made autograd, C++ stdlib).
add_executable(dqn autograd.cpp main.cpp)

if (MSVC)
    target_compile_options(dqn PRIVATE /O2 /arch:AVX2 /fp:fast)
else()
    target_compile_options(dqn PRIVATE -O3 -march=native -funsafe-math-optimizations)
    find_package(Threads REQUIRED)
    target_link_libraries(dqn PRIVATE Threads::Threads)
endif()
