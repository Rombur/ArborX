find_package(caliper REQUIRED PATHS ${CALIPER_DIR})
add_executable(ArborX_SurfaceReconstruction.exe example_surface_reconstruction.cpp)
target_link_libraries(ArborX_SurfaceReconstruction.exe ArborX::ArborX Boost::program_options caliper)
add_test(NAME ArborX_SurfaceReconstruction_Example COMMAND ./ArborX_SurfaceReconstruction.exe)
