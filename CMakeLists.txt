cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS "-fPIC -std=c++14 -DDEBUG")
set(CMAKE_C_FLAGS "-fPIC -std=c11 -DDEBUG")
project(LightTrack)

set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("OpenMP not found - continuing without it")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(LightTrack main.cpp LightTrack.cpp)

# Use static linking and specify library search paths
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${PROJECT_SOURCE_DIR}/lib")
target_link_libraries(LightTrack 
    ${PROJECT_SOURCE_DIR}/lib/libopencv_world.so 
    ${PROJECT_SOURCE_DIR}/lib/libncnn.so 
    ${PROJECT_SOURCE_DIR}/lib/libvulkan.so
    -ldl -lpthread)

## install target
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/lighttrack_demo)
install(TARGETS LightTrack RUNTIME DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(DIRECTORY lib/ DESTINATION ./)