/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_Ray.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

ArborX::Point read_point(char *facet)
{
  char f1[4] = {facet[0], facet[1], facet[2], facet[3]};
  char f2[4] = {facet[4], facet[5], facet[6], facet[7]};
  char f3[4] = {facet[8], facet[9], facet[10], facet[11]};

  ArborX::Point vertex;
  vertex[0] = *(reinterpret_cast<float *>(f1));
  vertex[1] = *(reinterpret_cast<float *>(f2));
  vertex[2] = *(reinterpret_cast<float *>(f3));

  return vertex;
}

// http://www.sgh1.net/posts/read-stl-file.md
template <typename MemorySpace>
Kokkos::View<ArborX::Point *[3], MemorySpace>
read_stl(std::string const &filename) {
  // TODO check that the file exists

  std::ifstream file(filename.c_str(), std::ios::binary);

  // read 80 byte header
  char header_info[80] = "";
  file.read(header_info, 80);

  // Read the number of Triangle
  unsigned int n_triangles = 0;
  {
    char n_tri[4];
    file.read(n_tri, 4);
    n_triangles = *(reinterpret_cast<unsigned long *>(*n_tri));
  }

  Kokkos::View<ArborX::Point *[3], MemorySpace> triangles(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"),
      n_triangles);

  auto triangles_host = Kokkos::create_mirror_view(triangles);

  for (unsigned int i = 0; i < n_triangles; ++i)
  {
    char facet[50];

    // read one 50-byte triangle
    file.read(facet, 50);

    // populate each point of the triangle
    // facet + 12 skips the triangle's unit normal
    auto p1 = read_point(facet + 12);
    auto p2 = read_point(facet + 24);
    auto p3 = read_point(facet + 36);

    // add a new triangle to the View
    triangles_host(i, 0) = p1;
    triangles_host(i, 1) = p2;
    triangles_host(i, 2) = p3;
  }
  file.close();

  Kokkos::deep_copy(triangles, triangles_host);

  return triangles;
}

template <typename MemorySpace>
Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> read_ray_file(
    std::string const &filename)
{
  // TODO check that the file exists
  std::ifstream file(filename.c_str());

  int n_rays = 0;
  file >> n_rays;

  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> rays(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "rays"), n_rays);

  auto rays_host = Kokkos::create_mirror_view(rays);

  for (int i = 0; i < n_rays; ++i)
  {
    float x, y, z, dir_x, dir_y, dir_z;
    file >> x >> y >> z >> dir_x >> dir_y >> dir_z;
    rays_host(i) = {{x, y, z}, {dir_x, dir_y, dir_z}};
  }
  file.close();

  Kokkos::deep_copy(rays, rays_host);

  return rays;
}

void outputTXT(Kokkos::View<ArborX::Point *, Kokkos::HostSpace> point_cloud,
               std::ostream &file, std::string const &delimiter)
{
  int const n_points = point_cloud.extent(0);
  for (int i = 0; i < n_points; ++i)
  {
    file << point_cloud(i)[0] << delimiter << point_cloud(i)[1] << delimiter
         << point_cloud(i)[2] << "\n";
  }
}

void outputVTK(Kokkos::View<ArborX::Point *, Kokkos::HostSpace> point_cloud,
               std::ostream &file)
{
  // Write the header
  file << "# vtk DataFile Version 2.0\n";
  file << "Ray tracing\n";
  file << "ASCII\n";
  file << "DATASET POLYDATA\n";

  int const n_points = point_cloud.extent(0);
  file << "POINTS " << n_points << " float\n";
  for (int i = 0; i < n_points; ++i)
  {
    file << point_cloud(i)[0] << " " << point_cloud(i)[1] << " "
         << point_cloud(i)[2] << "\n";
  }

  // We need to associate a value to each point. We arbitrarly choose 1.
  file << "POINTS_DATA " << n_points << "\n";
  file << "SCALARS value float 1\n";
  file << "LOOKUP_TABLE table\n";
  for (int i = 0; i < n_points; ++i)
  {
    file << "1.0\n";
  }
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;

  std::string stl_filename;
  std::string ray_filename;
  std::string output_filename;
  std::string output_type;
  int n_ray_files;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "help message" )
    ("stl_file,s", bpo::value<std::string>(&stl_filename), "name of the STL file")
    ("ray_files,r", bpo::value<std::string>(&ray_filename), "name of the ray file")
    ("n_ray_files,n", bpo::value<int>(&n_ray_files), "number of ray files")
    ("output_files,o", bpo::value<std::string>(&output_filename), "name of the output file")
    ("output_type,t", bpo::value<std::string>(&output_type), 
       "type of the output: csv, numpy, or vtk")
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  // Read the STL file
  auto triangles = read_stl<MemorySpace>(stl_filename);
  for (int i = 0; i < n_ray_files; ++i)
  {
    // Read the ray file
    std::string current_ray_filename =
        ray_filename + "-" + std::to_string(i) + ".txt";

    auto rays = read_ray_file<MemorySpace>(current_ray_filename);

    Kokkos::View<ArborX::Point *, MemorySpace> point_cloud("point_cloud", 10);
    auto point_cloud_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, point_cloud);

    // Write results to file
    if ((output_type == "csv") || (output_type == "numpy"))
    {
      std::string delimiter = " ";
      if (output_type == "csv")
        delimiter = ",";
      else
        delimiter = " ";
      std::ofstream file;
      file.open(output_filename + "-" + std::to_string(i) + ".txt");
      outputTXT(point_cloud_host, file, delimiter);
      file.close();
    }
    else
    {
      std::ofstream file;
      file.open(output_filename + "-" + std::to_string(i) + ".vtk");
      outputVTK(point_cloud_host, file);
      file.close();
    }
  }
}
