add_executable(ArborX_Projection.exe projection.cpp)
target_link_libraries(ArborX_Projection.exe ArborX::ArborX Boost::program_options)
add_test(NAME ArborX_Projection COMMAND ./ArborX_Projection)
