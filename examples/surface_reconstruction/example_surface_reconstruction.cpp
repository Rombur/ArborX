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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <math.h>
#include <pthread.h>

// clang-format off
// The algorithm is based on "Local Delaunay-based high fidelity surface
// reconstruction from 3D point sets" but it was simplified 
// The original algorithm is:
// Input: Input point set R^3
// Output: ReconstructedSurface R.
// 1: for each point p i in S do
// 2:   Compute the local neighborhood of p_i.
// 3:   Construct Delaunay triangulation, DT.
// 4:   Collect the triangles incident with p_i , T (p_i) (1-ring neighborhood of p_i in DT).
// 5:   Sort the triangles, T(p_i) in the ascending order of their circum-radii.
// 6:   for each triangle t_j in T (p_i) do
// 7:     if t_j is a prime-triangle complying with the non-overlapping triangle and manifold constraints then
// 8:       R = R union t_j.
// 9:     end if
// 10:  end for
// 11:end for
// 12:return R
// clang-format on

template <typename MemorySpace>
Kokkos::View<ArborX::Point *, MemorySpace> createPlane(int n_x, int n_y)
{
  CALI_CXX_MARK_FUNCTION;
  Kokkos::View<ArborX::Point *, MemorySpace> point_cloud(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "point_cloud"),
      n_x * n_y);
  auto point_cloud_host = Kokkos::create_mirror_view(point_cloud);
  for (int i = 0; i < n_x; ++i)
  {
    for (int j = 0; j < n_y; ++j)
    {
      point_cloud_host(j * n_y + i) = {2.f * i, 3.f * j, 0.f};
    }
  }
  Kokkos::deep_copy(point_cloud, point_cloud_host);

  return point_cloud;
}

template <typename MemorySpace>
Kokkos::View<ArborX::Point *, MemorySpace> createSphere(int n_points)
{
  CALI_CXX_MARK_FUNCTION;
  Kokkos::View<ArborX::Point *, MemorySpace> point_cloud(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "point_cloud"), n_points);
  auto point_cloud_host = Kokkos::create_mirror_view(point_cloud);
  std::uniform_real_distribution<float> uniform{-1.0, 1.0};
  std::default_random_engine gen;
  auto rand_uniform = [&]() { return uniform(gen); };

  std::uniform_real_distribution<float> uniform_z{0., 1.0};
  std::default_random_engine gen_z;
  auto rand_uniform_z = [&]() { return uniform_z(gen_z); };
  for (int i = 0; i < n_points; ++i)
  {
    auto x = rand_uniform();
    auto y = rand_uniform();
    auto z = rand_uniform_z();
    auto norm = std::sqrt(x * x + y * y + z * z);
    point_cloud_host(i) = {x / norm, y / norm, z / norm};
  }
  Kokkos::deep_copy(point_cloud, point_cloud_host);

  return point_cloud;
}

template <typename MemorySpace>
Kokkos::View<ArborX::Point *, MemorySpace> createUniformSphere(int n_points)
{
  CALI_CXX_MARK_FUNCTION;
  // We try to have n_points but we cannot guarantee that we will get exactly
  // that number.
  int n_angle_points = std::sqrt(static_cast<double>(n_points));
  std::vector<ArborX::Point> point_cloud_vec;
  for (int i = 0; i < n_angle_points; ++i)
  {
    float const delta_theta =
        static_cast<float>(i) / static_cast<float>(n_angle_points);
    float const cos_theta = std::cos((M_PI / 2.) * delta_theta);
    float const sin_theta = std::sin((M_PI / 2.) * delta_theta);
    for (int j = 0; j < n_angle_points; ++j)
    {
      float const delta_phi =
          static_cast<float>(j) / static_cast<float>(n_angle_points);
      float const cos_phi = std::cos((2. * M_PI) * delta_phi);
      float const sin_phi = std::sin((2. * M_PI) * delta_phi);
      point_cloud_vec.push_back(
          {cos_phi * sin_theta, sin_phi * sin_theta, cos_theta});
    }
  }
  // Update the number of points
  n_points = point_cloud_vec.size();

  Kokkos::View<ArborX::Point *, MemorySpace> point_cloud(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "point_cloud"), n_points);
  auto point_cloud_host = Kokkos::create_mirror_view(point_cloud);
  for (int i = 0; i < n_points; ++i)
  {
    point_cloud_host(i) = point_cloud_vec[i];
  }
  Kokkos::deep_copy(point_cloud, point_cloud_host);

  return point_cloud;
}

// Create all possible triangles. The problem is that it creates too many
// triangles which slows down everything.
template <typename MemorySpace, typename ExecutionSpace>
void createAllPossibleTriangles(
    Kokkos::View<ArborX::Point *, MemorySpace> point_cloud,
    Kokkos::View<int *, MemorySpace> indices,
    Kokkos::View<int *, MemorySpace> offsets,
    Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> &triangles,
    Kokkos::View<int *[3], MemorySpace> &triangle_vertices)
{
  CALI_CXX_MARK_FUNCTION;
  std::vector<ArborX::Experimental::Triangle> tri_vectors;
  std::set<std::tuple<int, int, int>> permutation_set;
  std::vector<std::tuple<int, int, int>> triangle_vertices_vec;
  for (unsigned int i = 0; i < offsets.extent(0) - 1; ++i)
  {
    for (int j = offsets(i); j < offsets(i + 1); ++j)
    {
      for (int k = j + 1; k < offsets(i + 1); ++k)
      {
        for (int m = k + 1; m < offsets(i + 1); ++m)
        {
          // FIXME Get ride of vertex permutations
          std::vector<int> to_sort = {indices(j), indices(k), indices(m)};
          std::sort(to_sort.begin(), to_sort.end());
          std::tuple<int, int, int> p = {to_sort[0], to_sort[1], to_sort[2]};
          if (permutation_set.count(p) == 0)
          {
            tri_vectors.push_back({point_cloud(indices(j)),
                                   point_cloud(indices(k)),
                                   point_cloud(indices(m))});
            permutation_set.insert(p);
            triangle_vertices_vec.push_back(p);
          }
        }
      }
    }
  }

  Kokkos::realloc(triangles, tri_vectors.size());
  Kokkos::realloc(triangle_vertices, tri_vectors.size());
  for (unsigned int i = 0; i < tri_vectors.size(); ++i)
  {
    triangles(i) = tri_vectors[i];
    triangle_vertices(i, 0) = std::get<0>(triangle_vertices_vec[i]);
    triangle_vertices(i, 1) = std::get<1>(triangle_vertices_vec[i]);
    triangle_vertices(i, 2) = std::get<2>(triangle_vertices_vec[i]);
  }
}

template <typename MemorySpace, typename ExecutionSpace>
void createDelaunayTriangles(
    Kokkos::View<ArborX::Point *, MemorySpace> point_cloud,
    Kokkos::View<int *, MemorySpace> indices,
    Kokkos::View<int *, MemorySpace> offsets,
    Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> &triangles,
    Kokkos::View<int *[3], MemorySpace> &triangle_vertices)
{
  CALI_CXX_MARK_FUNCTION;

  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned, K> Vb;
  typedef CGAL::Delaunay_triangulation_cell_base_3<K> Cb;
  typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
  typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Delaunay;
  typedef Delaunay::Point Point;

  std::vector<ArborX::Experimental::Triangle> tri_vectors;
  std::vector<std::tuple<int, int, int>> triangle_vertices_vec;
  unsigned int n_delaunay_triangles = 0;
  for (unsigned int i = 0; i < offsets.extent(0) - 1; ++i)
  {
    std::vector<std::pair<Point, int>> cgal_points;
    // FIXME Get ride of vertex permutations
    for (int j = offsets(i); j < offsets(i + 1); ++j)
    {
      // Add the vertex to cgal_points
      cgal_points.push_back(std::make_pair(Point(point_cloud(indices(j))[0],
                                                 point_cloud(indices(j))[1],
                                                 point_cloud(indices(j))[2]),
                                           indices(j)));
    }
    // Create the triangulation
    Delaunay triangulation(cgal_points.begin(), cgal_points.end());
    // Extract the relevant triangles and the indices
    for (auto v : triangulation.finite_cell_handles())
    {
      int index_a = v->vertex(0)->info();
      int index_b = v->vertex(1)->info();
      int index_c = v->vertex(2)->info();
      if ((index_a == (int)i) || (index_b == (int)i) || (index_c == (int)i))
      {
        tri_vectors.push_back(
            {point_cloud(index_a), point_cloud(index_b), point_cloud(index_c)});
        triangle_vertices_vec.push_back({index_a, index_b, index_c});
      }
      ++n_delaunay_triangles;
    }
  }
  std::cout << "Number of Delaunay triangles " << n_delaunay_triangles
            << std::endl;

  Kokkos::realloc(triangles, tri_vectors.size());
  Kokkos::realloc(triangle_vertices, tri_vectors.size());
  for (unsigned int i = 0; i < tri_vectors.size(); ++i)
  {
    triangles(i) = tri_vectors[i];
    triangle_vertices(i, 0) = std::get<0>(triangle_vertices_vec[i]);
    triangle_vertices(i, 1) = std::get<1>(triangle_vertices_vec[i]);
    triangle_vertices(i, 2) = std::get<2>(triangle_vertices_vec[i]);
  }
}

KOKKOS_INLINE_FUNCTION float
computeCircumRadius(ArborX::Experimental::Triangle triangle)
{
  CALI_CXX_MARK_FUNCTION;
  // https://mathworld.wolfram.com/Circumradius.html
  // (a*b*c) / sqrt((a+b+c)*(b+c-a)*(a+c-b)*(a+b-c))
  float a_sqr = 0;
  float b_sqr = 0;
  float c_sqr = 0;
  for (int i = 0; i < 3; ++i)
  {
    a_sqr += (triangle.a[i] - triangle.b[i]) * (triangle.a[i] - triangle.b[i]);
    b_sqr += (triangle.b[i] - triangle.c[i]) * (triangle.b[i] - triangle.c[i]);
    c_sqr += (triangle.c[i] - triangle.a[i]) * (triangle.c[i] - triangle.a[i]);
  }
  float a = std::sqrt(a_sqr);
  float b = std::sqrt(b_sqr);
  float c = std::sqrt(c_sqr);
  float const det =
      std::sqrt((a + b + c) * (b + c - a) * (a + c - b) * (a + b - c));

  // Not all the triangles in the triangle View are actually triangles. Some
  // elements are just three points aligned, in that case det is zero. To avoid
  // the division by zero, we just set the radius to a large number.
  return std::abs(det) < 1e-12 ? 1e12 : (a * b * c) / det;
}

template <typename MemorySpace, typename ExecutionSpace>
void sortTriangles(
    Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> &triangles,
    Kokkos::View<int *[3], MemorySpace> &triangle_vertices)
{
  CALI_CXX_MARK_FUNCTION;
  int const n_triangles = triangles.extent(0);
  Kokkos::View<float *, MemorySpace> radius("radius", n_triangles);
  Kokkos::parallel_for(
      "AborX::Surface::compute_radius",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_triangles),
      KOKKOS_LAMBDA(int i) { radius(i) = computeCircumRadius(triangles(i)); });

  // Sort the radius. If we want to follow the paper, we should not use a global
  // sort of the radius. Instead, we should only sort the triangles that belong
  // to the same query. This way, we would work one point of the point cloud at
  // the time. Instead, we perform the sort on all the radii to increase
  // parallelism.
  ExecutionSpace exec{};
  auto permutation = ArborX::Details::sortObjects(exec, radius);

  // Apply the permutation
  ArborX::Details::applyPermutation(exec, permutation, triangles);
  ArborX::Details::applyPermutation(exec, permutation, triangle_vertices);

  // Resize the view to remove degenerated triangles
  int new_size = -1;
  auto radius_host = Kokkos::create_mirror_view(radius);
  Kokkos::deep_copy(radius_host, radius);
  for (int i = 0; i < n_triangles; ++i)
  {
    if (radius_host(i) > 1e10)
    {
      new_size = i;
      break;
    }
  }
  if (new_size >= 0)
  {
    Kokkos::resize(triangles, new_size);
    Kokkos::resize(triangle_vertices, new_size);
  }

  std::cout << "Number of triangles after filtering " << triangles.extent(0)
            << std::endl;
}

template <typename MemorySpace, typename ExecutionSpace>
bool overlappingTriangles(ArborX::Experimental::Triangle const &tri_surface,
                          ArborX::Experimental::Triangle const &triangle)
{
  CALI_CXX_MARK_FUNCTION;
  // Create two rays for each edge of the triangle
  CALI_MARK_BEGIN("build ray/edges");
  std::vector<ArborX::Experimental::Ray> rays;
  rays.emplace_back(ArborX::Experimental::Ray{
      triangle.a, ArborX::Experimental::makeVector(triangle.a, triangle.b)});
  rays.emplace_back(ArborX::Experimental::Ray{
      triangle.b, ArborX::Experimental::makeVector(triangle.b, triangle.a)});
  rays.emplace_back(ArborX::Experimental::Ray{
      triangle.a, ArborX::Experimental::makeVector(triangle.a, triangle.c)});
  rays.emplace_back(ArborX::Experimental::Ray{
      triangle.c, ArborX::Experimental::makeVector(triangle.c, triangle.a)});
  rays.emplace_back(ArborX::Experimental::Ray{
      triangle.b, ArborX::Experimental::makeVector(triangle.b, triangle.c)});
  rays.emplace_back(ArborX::Experimental::Ray{
      triangle.c, ArborX::Experimental::makeVector(triangle.c, triangle.b)});

  float epsilon = 1e-6;
  std::vector<float> edge_lengths(6);
  float ab = std::sqrt(
      (triangle.a[0] - triangle.b[0]) * (triangle.a[0] - triangle.b[0]) +
      (triangle.a[1] - triangle.b[1]) * (triangle.a[1] - triangle.b[1]) +
      (triangle.a[2] - triangle.b[2]) * (triangle.a[2] - triangle.b[2]));
  float ac = std::sqrt(
      (triangle.a[0] - triangle.c[0]) * (triangle.a[0] - triangle.c[0]) +
      (triangle.a[1] - triangle.c[1]) * (triangle.a[1] - triangle.c[1]) +
      (triangle.a[2] - triangle.c[2]) * (triangle.a[2] - triangle.c[2]));
  float bc = std::sqrt(
      (triangle.b[0] - triangle.c[0]) * (triangle.b[0] - triangle.c[0]) +
      (triangle.b[1] - triangle.c[1]) * (triangle.b[1] - triangle.c[1]) +
      (triangle.b[2] - triangle.c[2]) * (triangle.b[2] - triangle.c[2]));
  edge_lengths[0] = ab;
  edge_lengths[1] = ab;
  edge_lengths[2] = ac;
  edge_lengths[3] = ac;
  edge_lengths[4] = bc;
  edge_lengths[5] = bc;
  CALI_MARK_END("build ray/edges");

  CALI_MARK_BEGIN("compute intersection ray/edges");
  for (int i = 0; i < 6; ++i)
  {
    float t_min;
    float t_max;
    if (ArborX::Experimental::intersection(rays[i], tri_surface, t_min, t_max))
    {
      if ((epsilon * edge_lengths[i] < t_min) &&
          (t_min < (1.f - epsilon) * edge_lengths[i]))
      {
        CALI_MARK_END("compute intersection ray/edges");
        return true;
      }
    }
  }

  CALI_MARK_END("compute intersection ray/edges");
  return false;
}

void commonEdge(std::array<int, 3> const &surface_triangle_indices,
                std::array<int, 3> const &triangle_indices,
                std::vector<int> &edges)
{
  CALI_CXX_MARK_FUNCTION;
  int tri_a = triangle_indices[0];
  int tri_b = triangle_indices[1];
  int tri_c = triangle_indices[2];
  int surface_tri_a = surface_triangle_indices[0];
  int surface_tri_b = surface_triangle_indices[1];
  int surface_tri_c = surface_triangle_indices[2];

  // ab = 0
  // bc = 1
  // ca = 2

  if ((tri_a == surface_tri_a) || (tri_a == surface_tri_b) ||
      (tri_a == surface_tri_c))
  {
    if ((tri_b == surface_tri_a) || (tri_b == surface_tri_b) ||
        (tri_b == surface_tri_c))
    {
      edges[0] += 1;
    }
    if ((tri_c == surface_tri_a) || (tri_c == surface_tri_b) ||
        (tri_c == surface_tri_c))
    {
      edges[2] += 1;
    }
  }
  if ((tri_b == surface_tri_a) || (tri_b == surface_tri_b) ||
      (tri_b == surface_tri_c))
  {
    if ((tri_c == surface_tri_a) || (tri_c == surface_tri_b) ||
        (tri_c == surface_tri_c))
    {
      edges[1] += 1;
    }
  }
}

bool satisfyManifold( // int const triangle_surface_indices[3],
    std::array<int, 3> const &triangle_surface_indices,
    std::array<int, 3> const &triangle_indices, std::vector<int> &edges)
{
  CALI_CXX_MARK_FUNCTION;
  // Check that each edge is shared by at most two triangles.
  commonEdge(triangle_surface_indices, triangle_indices, edges);
  for (int e : edges)
  {
    if (e > 2)
    {
      return false;
    }
  }

  return true;
}

template <typename MemorySpace, typename ExecutionSpace>
Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> createSurface(
    Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> triangles,
    Kokkos::View<int *[3], MemorySpace> triangle_vertices, bool check_overlap,
    bool check_manifold)
{
  CALI_CXX_MARK_FUNCTION;
  CALI_MARK_BEGIN("Surface nearest neighbors search");
  float const eps = 0.01;
  ExecutionSpace exec_space{};
  Kokkos::View<ArborX::Box *, MemorySpace> boxes("boxes", triangles.extent(0));
  Kokkos::parallel_for(
      "AborX::Surface::expand_triangles",
      Kokkos::RangePolicy<ExecutionSpace>(0, triangles.extent(0)),
      KOKKOS_LAMBDA(int i) {
        auto const &a = triangles(i).a;
        auto const &b = triangles(i).b;
        auto const &c = triangles(i).c;
        ArborX::Point min_corner = {std::min(std::min(a[0], b[0]), c[0]),
                                    std::min(std::min(a[1], b[1]), c[1]),
                                    std::min(std::min(a[2], b[2]), c[2])};
        for (int d = 0; d < 3; ++d)
          min_corner[d] -= eps * std::abs(min_corner[d]);
        ArborX::Point max_corner = {
            (1 + eps) * std::max(std::max(a[0], b[0]), c[0]),
            (1 + eps) * std::max(std::max(a[1], b[1]), c[1]),
            (1 + eps) * std::max(std::max(a[2], b[2]), c[2])};
        boxes(i) = ArborX::Box{min_corner, max_corner};
      });
  Kokkos::View<ArborX::Intersects<ArborX::Box> *, MemorySpace> queries(
      "boxes", triangles.extent(0));
  Kokkos::parallel_for(
      "AborX::Surface::intersect_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, triangles.extent(0)),
      KOKKOS_LAMBDA(int i) { queries(i) = ArborX::intersects(boxes(i)); });
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
  ArborX::BVH<MemorySpace> const bvh(exec_space, boxes);
  bvh.query(exec_space, queries, indices, offsets);
  CALI_MARK_END("Surface nearest neighbors search");

  // Add triangle to the surface
  CALI_MARK_BEGIN("Surface add to surface");
  auto triangles_host = Kokkos::create_mirror_view(triangles);
  Kokkos::deep_copy(triangles_host, triangles);
  auto triangle_vertices_host = Kokkos::create_mirror_view(triangle_vertices);
  Kokkos::deep_copy(triangle_vertices_host, triangle_vertices);
  std::vector<ArborX::Experimental::Triangle> surface_vec;
  std::vector<bool> surface_indices(triangles.extent(0), false);
  unsigned int n_overlap = 0;
  unsigned int n_manifold = 0;
  for (unsigned int i = 0; i < triangles_host.extent(0); ++i)
  {
    bool add_triangle = true;
    auto triangle = triangles_host(i);
    std::vector<int> edges = {0, 0, 0};
    for (int k = offsets(i); k < offsets(i + 1); ++k)
    {
      // Only compared to triangles that are part of the surface
      if (surface_indices[indices(k)] == true)
      {
        // Check overlapping
        if (check_overlap && overlappingTriangles<MemorySpace, ExecutionSpace>(
                                 triangles_host(indices[k]), triangle))
        {
          add_triangle = false;
          ++n_overlap;
          break;
        }

        // Check manifold constraints
        if (check_manifold &&
            !satisfyManifold({triangle_vertices_host(indices[k], 0),
                              triangle_vertices_host(indices[k], 1),
                              triangle_vertices_host(indices[k], 2)},
                             {triangle_vertices_host(i, 0),
                              triangle_vertices_host(i, 1),
                              triangle_vertices_host(i, 2)},
                             edges))
        {
          add_triangle = false;
          ++n_manifold;
          break;
        }
      }
    }
    if (add_triangle)
    {
      surface_vec.push_back(triangle);
      surface_indices[i] = true;
    }
  }
  CALI_MARK_END("Surface add to surface");

  std::cout << "Triangles rejected: overlap " << n_overlap << std::endl;
  std::cout << "Triangles rejected: manifold " << n_manifold << std::endl;

  Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> surface(
      "surface", surface_vec.size());
  auto surface_host = Kokkos::create_mirror_view(surface);
  for (unsigned int i = 0; i < surface_vec.size(); ++i)
  {
    surface_host(i) = surface_vec[i];
  }
  Kokkos::deep_copy(surface, surface_host);

  return surface;
}

template <typename MemorySpace>
void output(Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> surface,
            std::ostream &file)
{
  CALI_CXX_MARK_FUNCTION;
  auto surface_host = Kokkos::create_mirror_view(surface);
  Kokkos::deep_copy(surface_host, surface);
  int const n_cells = surface.extent(0);
  std::vector<std::vector<float>> vertices;
  std::vector<std::vector<int>> cells(n_cells);
  for (int i = 0; i < n_cells; ++i)
  {
    vertices.push_back({surface(i).a[0], surface(i).a[1], surface(i).a[2]});
    vertices.push_back({surface(i).b[0], surface(i).b[1], surface(i).b[2]});
    vertices.push_back({surface(i).c[0], surface(i).c[1], surface(i).c[2]});
    cells[i] = {3 * i, 3 * i + 1, 3 * i + 2};
  }

  int const n_points = vertices.size();

  // Write the header
  file << "# vtk DataFile Version 2.0\n";
  file << "Surface Reconstruction\n";
  file << "ASCII\n";
  file << "DATASET UNSTRUCTURED_GRID\n";

  file << "POINTS " << n_points << " float\n";
  for (int i = 0; i < n_points; ++i)
  {
    file << vertices[i][0] << " " << vertices[i][1] << " " << vertices[i][2]
         << "\n";
  }

  file << "CELLS " << n_cells << " " << 4 * n_cells << "\n";
  for (int i = 0; i < n_cells; ++i)
  {
    file << "3 " << cells[i][0] << " " << cells[i][1] << " " << cells[i][2]
         << "\n";
  }

  file << "CELL_TYPES " << n_cells << "\n";
  for (int i = 0; i < n_cells; ++i)
  {
    file << 5 << "\n";
  }

  // We need to associate a value to each point. We arbitrarly choose 1.
  file << "POINT_DATA " << n_points << "\n";
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

  cali::ConfigManager caliper_manager;
  // caliper_manager.add(argv[1]);
  caliper_manager.start();
  CALI_CXX_MARK_FUNCTION;

  int n_neighbors;
  int n_samples;
  std::string point_cloud_type;
  std::string triangulation_type;
  bool check_overlap;
  bool check_manifold;

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ( "help", "help message" )
    ("k", bpo::value<int>(&n_neighbors)->default_value(10), "number of neighbors")
    ("n", bpo::value<int>(&n_samples)->default_value(10000), "number of points")
    ("g", bpo::value<std::string>(&point_cloud_type)->default_value("sphere"), 
     "geometry of the domain")
    ("t", bpo::value<std::string>(&triangulation_type)->default_value("delaunay"), 
     "triangulation type")
    ("o", bpo::value<bool>(&check_overlap)->default_value(true), "check overlap")
    ("m", bpo::value<bool>(&check_manifold)->default_value(true), "check manifold")
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

  // Create a points in a plane
  Kokkos::View<ArborX::Point *, MemorySpace> point_cloud;
  if (point_cloud_type == "plane")
  {
    int n = std::sqrt(static_cast<double>(n_samples));
    point_cloud = createPlane<MemorySpace>(n, n);
  }
  else if (point_cloud_type == "sphere")
  {
    point_cloud = createSphere<MemorySpace>(n_samples);
  }
  else
  {
    point_cloud = createUniformSphere<MemorySpace>(n_samples);
  }
  int const n_points = point_cloud.extent(0);
  std::cout << "Number of points " << n_points << std::endl;
  std::cout << "Number of nearest neighbors " << n_neighbors << std::endl;

  ExecutionSpace exec_space{};

  // First we find the local neighborhood
  CALI_MARK_BEGIN("nearest neighbors search");
  Kokkos::View<ArborX::Nearest<ArborX::Point> *, MemorySpace> queries("queries",
                                                                      n_points);
  Kokkos::parallel_for(
      "AborX::Surface::query", Kokkos::RangePolicy<ExecutionSpace>(0, n_points),
      KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::nearest(point_cloud(i), n_neighbors);
      });
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("offsets", 0);

  ArborX::BVH<MemorySpace> bvh{exec_space, point_cloud};
  bvh.query(exec_space, queries, indices, offsets);
  CALI_MARK_END("nearest neighbors search");

  // Now we are supposed to create a Delaunay triangulation. This is not trivial
  // to do. Instead, we create every possible triangle. This means that later we
  // will have more possible candidates and the quality of the reconstruction
  // will be degraded but it simplifies the algorithm.
  Kokkos::View<ArborX::Experimental::Triangle *, MemorySpace> triangles(
      "triangles", 0);
  Kokkos::View<int *[3], MemorySpace> triangle_vertices("triangle_vertices", 0);
  if (triangulation_type == "delaunay")
  {
    createDelaunayTriangles<MemorySpace, ExecutionSpace>(
        point_cloud, indices, offsets, triangles, triangle_vertices);
  }
  else
  {
    createAllPossibleTriangles<MemorySpace, ExecutionSpace>(
        point_cloud, indices, offsets, triangles, triangle_vertices);
  }
  std::cout << "Number of triangles created " << triangles.extent(0)
            << std::endl;

  // Sort the triangle by circum-radius
  sortTriangles<MemorySpace, ExecutionSpace>(triangles, triangle_vertices);

  // Create the surface
  auto surface = createSurface<MemorySpace, ExecutionSpace>(
      triangles, triangle_vertices, check_overlap, check_manifold);
  std::cout << "Number of triangles in the surface " << surface.size()
            << std::endl;

  // Output the surface
  std::ofstream file;
  file.open("surface.vtk");
  output(surface, file);
  file.close();

  caliper_manager.flush();

  return 0;
}
