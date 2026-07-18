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

# Phase 3: DQN x Super Mario (RAM features). Reuses the DQN core + the NES core.
add_executable(mario_dqn mario_dqn.cpp mario.cpp nes.cpp autograd.cpp ${LAINES_SRC})
target_include_directories(mario_dqn PRIVATE . third_party/laines/include)
if (MSVC)
    target_compile_options(mario_dqn PRIVATE /O2 /permissive-)
else()
    target_compile_options(mario_dqn PRIVATE -O2)
    find_package(Threads REQUIRED)
    target_link_libraries(mario_dqn PRIVATE Threads::Threads)
endif()
# The CPU-throttle uses timeBeginPeriod on Windows (guarded by #ifdef _WIN32),
# which lives in winmm. _WIN32 is defined by ALL Windows toolchains (MSVC AND
# MinGW/w64devkit), so link winmm for any Windows build -- not just MSVC. On
# Linux/macOS/Emscripten _WIN32 is undefined and the throttle uses portable
# std::this_thread::sleep_for, so winmm must NOT be linked there.
if (WIN32)
    target_link_libraries(mario_dqn PRIVATE winmm)
endif()

# 3-action "human-strategy" variant (RUN / RUNJUMP / STOMP). Same sources, built
# with -DMARIO_ACTIONS=3. From scratch, exploration reaches a full 1-1 clear far
# sooner than the 5-action set (see RESUME). Nets/demos are action-set-specific.
add_executable(mario_dqn3 mario_dqn.cpp mario.cpp nes.cpp autograd.cpp ${LAINES_SRC})
target_include_directories(mario_dqn3 PRIVATE . third_party/laines/include)
target_compile_definitions(mario_dqn3 PRIVATE MARIO_ACTIONS=3)
if (MSVC)
    target_compile_options(mario_dqn3 PRIVATE /O2 /permissive-)
else()
    target_compile_options(mario_dqn3 PRIVATE -O2)
    find_package(Threads REQUIRED)
    target_link_libraries(mario_dqn3 PRIVATE Threads::Threads)
endif()
if (WIN32)
    target_link_libraries(mario_dqn3 PRIVATE winmm)
endif()

# Phase 4: DQN x Super Mario 1-2 (underground). Separate trainer/env from 1-1,
# sharing the level-agnostic core (mario_shared.h). Warm-starts from a 1-1 net.
add_executable(mario12_dqn train12.cpp mario12.cpp nes.cpp autograd.cpp ${LAINES_SRC})
target_include_directories(mario12_dqn PRIVATE . third_party/laines/include)
if (MSVC)
    target_compile_options(mario12_dqn PRIVATE /O2 /permissive-)
else()
    target_compile_options(mario12_dqn PRIVATE -O2)
    find_package(Threads REQUIRED)
    target_link_libraries(mario12_dqn PRIVATE Threads::Threads)
endif()
if (WIN32)
    target_link_libraries(mario12_dqn PRIVATE winmm)
endif()

# World 1-3 / 1-4 trainers (warp-started; share the generic templated trainer).
add_executable(mario13_dqn train13.cpp mario13.cpp nes.cpp autograd.cpp ${LAINES_SRC})
add_executable(mario14_dqn train14.cpp mario14.cpp nes.cpp autograd.cpp ${LAINES_SRC})
add_executable(mario13_rec record13.cpp mario13.cpp nes.cpp autograd.cpp ${LAINES_SRC})
add_executable(mario14_rec record14.cpp mario14.cpp nes.cpp autograd.cpp ${LAINES_SRC})
foreach(t mario13_dqn mario14_dqn mario13_rec mario14_rec)
    target_include_directories(${t} PRIVATE . third_party/laines/include)
    if (MSVC)
        target_compile_options(${t} PRIVATE /O2 /permissive-)
    else()
        target_compile_options(${t} PRIVATE -O2)
        find_package(Threads REQUIRED)
        target_link_libraries(${t} PRIVATE Threads::Threads)
    endif()
    if (WIN32)
        target_link_libraries(${t} PRIVATE winmm)
    endif()
endforeach()

# 1-2 agent video recorder (separate exe so it runs while workers train).
add_executable(mario12_rec record12.cpp mario12.cpp nes.cpp autograd.cpp ${LAINES_SRC})
target_include_directories(mario12_rec PRIVATE . third_party/laines/include)
if (MSVC)
    target_compile_options(mario12_rec PRIVATE /O2 /permissive-)
else()
    target_compile_options(mario12_rec PRIVATE -O2)
endif()

# Static file server for the results viewer (cpp-httplib, single-header MIT).
add_executable(server server.cpp)
target_include_directories(server PRIVATE third_party)
if (MSVC)
    target_compile_options(server PRIVATE /O2)
    target_link_libraries(server PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(server PRIVATE Threads::Threads)
endif()
