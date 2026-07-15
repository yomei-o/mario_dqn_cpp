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

# Phase 2: headless NES (vendored LaiNES core, SDL/audio stripped) + boot test.
file(GLOB LAINES_SRC third_party/laines/*.cpp third_party/laines/mappers/*.cpp)
add_executable(nes_test nes_test.cpp nes.cpp ${LAINES_SRC})
target_include_directories(nes_test PRIVATE . third_party/laines/include)
# The vendored LaiNES core was de-GNU-ified (case-ranges -> if/else) so it builds
# with plain MSVC. /permissive- enables the iso646 alternative tokens (and/or/not).
if (MSVC)
    target_compile_options(nes_test PRIVATE /O2 /permissive-)
else()
    target_compile_options(nes_test PRIVATE -O2)
endif()
