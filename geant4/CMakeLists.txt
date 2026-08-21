cmake_minimum_required(VERSION 3.16)
project(mstSim LANGUAGES CXX)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(mstSim
  mstSim.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/PhysicsListFactory.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SteppingAction.cc
)

target_include_directories(mstSim PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mstSim PRIVATE ${Geant4_LIBRARIES})
target_compile_features(mstSim PRIVATE cxx_std_17)

# Put the executable in the build directory as ./mstSim.
set_target_properties(mstSim PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
