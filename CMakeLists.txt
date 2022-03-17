cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS "-fPIC -std=c++14 -DDEBUG")
set(CMAKE_C_FLAGS "-fPIC -std=c11 -DDEBUG")
project(LightTrack)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include /home/pxierra/Downloads/vulkansdk-linux-x86_64-1.3.204.1/1.3.204.1/source)

link_directories(${PROJECT_SOURCE_DIR}/lib /home/pxierra/Downloads/vulkansdk-linux-x86_64-1.3.204.1/1.3.204.1/x86_64/lib)

add_executable(LightTrack main.cpp LightTrack.cpp)

target_link_libraries(LightTrack -lopencv_world libncnn.a -lvulkan -lglslang -lshaderc_shared)

## install target
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/lighttrack_demo)
install(TARGETS LightTrack RUNTIME DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(DIRECTORY lib/ DESTINATION ./)