cmake_minimum_required(VERSION 3.16)
project(mstSim)

# Requires Geant4 11.x (uses G4RunManagerFactory, standard since 10.7).
# Build:
#   mkdir build && cd build
#   cmake -DGeant4_DIR=/path/to/geant4/install/lib/Geant4-11.x ..
#   make -j4
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(mstSim mstSim.cc ${sources} ${headers})
target_link_libraries(mstSim ${Geant4_LIBRARIES})
