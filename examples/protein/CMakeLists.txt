cmake_minimum_required(VERSION 3.20)

find_package(HDF5 COMPONENTS CXX)
add_executable(ArborX_Protein.exe example_protein.cpp)
target_link_libraries(ArborX_Protein.exe ArborX::ArborX HDF5::HDF5)
add_test(NAME ArborX_Protein_Example COMMAND ./ArborX_Protein.exe)
