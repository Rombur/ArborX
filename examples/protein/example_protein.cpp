/****************************************************************************
 * Copyright (c) 2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_ViewCtor.hpp>

#include <fstream>
#include <string>
#include <tuple>

#include <H5Cpp.h>

using coord_datatype = std::vector<std::vector<std::vector<float>>>;

std::vector<int> readIndices(std::string const &filename)
{
  std::ifstream file(filename);
  std::vector<int> indices;
  if (file.is_open())
  {
    while (file.good())
    {
      double index = -1.;
      file >> index;
      if (index == -1)
        break;
      indices.push_back(index);
    }
  }
  return indices;
}

std::tuple<coord_datatype, std::vector<int>, std::vector<int>>
readData(std::string const &filename)
{
  // Open HDF5 file read only
  H5::H5File file(filename, H5F_ACC_RDONLY);

  // Open the coordinates data set
  H5::DataSet coord_dataset = file.openDataSet("coordinates");
  // Get the DataSpace
  H5::DataSpace coord_dataspace = coord_dataset.getSpace();
  // Check the number of dimensions of the data
  assert(coord_dataspace.getSimpleExtentNdims() == 3);
  // Read the dimensions of the data: frame, n points, dim
  hsize_t dims[3];
  coord_dataspace.getSimpleExtentDims(dims);
  int n_frames = dims[0];
  int n_points = dims[1];
  assert(dims[2] == 3);
  std::cout << "n frames " << n_frames << "\nn points " << n_points
            << std::endl;
  // Check the dataset type
  assert(coord_dataset.getTypeClass() == H5T_FLOAT);
  assert(coord_dataset.getFloatType().getSize() == 4);
  // Read the data
  H5::DataSpace coord_memspace(3, dims);
  std::vector<float> coord_vector(n_frames * n_points * 3);
  coord_dataset.read(coord_vector.data(), H5::PredType::NATIVE_FLOAT,
                     coord_memspace, coord_dataspace);
  // Reshape the data
  coord_datatype coordinates(n_frames, std::vector<std::vector<float>>(
                                           n_points, std::vector<float>(3)));
  for (int i = 0; i < n_frames; ++i)
  {
    for (int j = 0; j < n_points; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        coordinates[i][j][k] = coord_vector[i * 3 * n_points + j * 3 + k];
      }
    }
  }

  // Now we read the indices of the 'protein' and the 'not protein'
  auto protein_indices = readIndices("protein_indices.txt");
  std::cout << "n protein indices " << protein_indices.size() << std::endl;
  auto not_protein_indices = readIndices("not_protein_indices.txt");
  std::cout << "n not_protein indices " << not_protein_indices.size()
            << std::endl;

  return std::make_tuple(coordinates, protein_indices, not_protein_indices);
}

template <typename ExecutionSpace, typename MemorySpace>
std::tuple<Kokkos::View<ArborX::Point *, MemorySpace>,
           Kokkos::View<ArborX::Point *, MemorySpace>>
convertData(ExecutionSpace const &execution_space, int frame,
            coord_datatype const &coordinates,
            std::vector<int> const &query_indices,
            std::vector<int> const &database_indices)
{
  unsigned int const n_query_pts = query_indices.size();
  Kokkos::View<ArborX::Point *, MemorySpace> query_pts(
      Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                         "query_points"),
      n_query_pts);
  auto query_pts_host = Kokkos::create_mirror_view(query_pts);
  for (unsigned int i = 0; i < n_query_pts; ++i)
  {
    query_pts_host(i) = {coordinates[frame][query_indices[i]][0],
                         coordinates[frame][query_indices[i]][1],
                         coordinates[frame][query_indices[i]][2]};
  }
  Kokkos::deep_copy(execution_space, query_pts, query_pts_host);

  unsigned int const n_database_pts = database_indices.size();
  Kokkos::View<ArborX::Point *, MemorySpace> database_pts(
      Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                         "query_points"),
      n_database_pts);
  auto database_pts_host = Kokkos::create_mirror_view(database_pts);
  for (unsigned int i = 0; i < n_database_pts; ++i)
  {
    database_pts_host(i) = {coordinates[frame][database_indices[i]][0],
                            coordinates[frame][database_indices[i]][1],
                            coordinates[frame][database_indices[i]][2]};
  }
  Kokkos::deep_copy(execution_space, database_pts, database_pts_host);

  execution_space.fence();

  return std::make_tuple(query_pts, database_pts);
}

template <typename ViewType>
void writeRawResults(ViewType indices, ViewType offsets, std::ofstream &file)
{
  for (unsigned int i = 0; i < indices.extent(0); ++i)
  {
    file << indices(i) << " ";
  }
  file << "\n";

  for (unsigned int i = 0; i < offsets.extent(0); ++i)
  {
    file << offsets(i) << " ";
  }
  file << "\n";
}

void writeResults(std::vector<std::vector<int>> const &results,
                  std::ofstream &file)
{
  for (auto const &res : results)
  {
    for (auto const val : res)
    {
      file << val << " ";
    }
    file << "\n";
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  ExecutionSpace execution_space;

  std::string filename = "prolint.h5";
  std::string output_filename = "arborx.txt";
  std::ofstream output_file(output_filename);
  std::string rawoutput_filename = "arborx_raw.txt";
  std::ofstream rawoutput_file(rawoutput_filename);

  // Read coordinate of all the points for all the frames
  auto [coordinates, protein_indices, not_protein_indices] = readData(filename);

  int n_frames_of_interest = coordinates.size(); // 10;
  float radius = 0.7;
  std::vector<Kokkos::View<ArborX::Point *, MemorySpace>> query_pts_vec;
  std::vector<Kokkos::View<ArborX::Point *, MemorySpace>> database_pts_vec;
  for (int f = 0; f < n_frames_of_interest; ++f)
  {
    auto [query_pts, database_pts] = convertData<ExecutionSpace, MemorySpace>(
        execution_space, f, coordinates, protein_indices, not_protein_indices);
    query_pts_vec.push_back(query_pts);
    database_pts_vec.push_back(database_pts);
  }
  std::cout << "Done reading the input files" << std::endl;

  for (int f = 0; f < n_frames_of_interest; ++f)
  {
    Kokkos::Profiling::pushRegion("ProLint::search");
    auto query_pts_view = query_pts_vec[f];
    unsigned int const n_queries = query_pts_view.extent(0);
    Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, MemorySpace>
        within_queries("within_queries", n_queries);
    Kokkos::parallel_for(
        "Setup queries",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          within_queries(i) =
              ArborX::intersects(ArborX::Sphere{query_pts_view(i), radius});
        });

    ArborX::BVH<MemorySpace> bvh(execution_space, database_pts_vec[f]);
    Kokkos::View<int *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    bvh.query(execution_space, within_queries, indices, offsets);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("ProLint::postprocessing");
    // Move the data to the host
    auto indices_host = Kokkos::create_mirror_view(indices);
    Kokkos::deep_copy(indices_host, indices);
    auto offsets_host = Kokkos::create_mirror_view(offsets);
    Kokkos::deep_copy(offsets_host, offsets);
    writeRawResults(indices_host, offsets_host, rawoutput_file);
    // Reformat results
    std::vector<std::vector<int>> results;
    for (unsigned int i = 0; i < offsets_host.extent(0) - 1; ++i)
    {
      std::vector<int> single_query_results;
      for (int j = offsets_host(i); j < offsets_host(i + 1); ++j)
      {
        single_query_results.push_back(indices_host(j));
      }
      // Make sure that we always have an entry for the query
      if (single_query_results.size() == 0)
      {
        single_query_results.push_back(-1);
      }
      results.push_back(single_query_results);
    }
    // Write results to file
    writeResults(results, output_file);
    Kokkos::Profiling::popRegion();
  }

  output_file.close();

  return 0;
}
