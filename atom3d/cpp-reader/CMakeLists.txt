cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hdf)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(readhdf readhdf.cpp)
target_link_libraries(readhdf ${HDF5_HL_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
set_property(TARGET readhdf PROPERTY CXX_STANDARD 17)

target_include_directories (readhdf PRIVATE "${HDF5_SRC_DIR};${HDF5_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
target_link_libraries (readhdf PRIVATE ${HDF5_CPP_LIB_TARGET} ${HDF5_LIB_TARGET})

message(STATUS "INCLUDE LOCATION" ${HDF5_INCLUDE_DIRS})
message(STATUS "version" ${HDF5_VERSION})
message(STATUS "DEFINITIONS" ${HDF5_DEFINITIONS})
message(STATUS "LIBRARIES" ${HDF5_LIBRARIES})
message(STATUS "HL_LIBRARIES" ${HDF5_HL_LIBRARIES})

