/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <random>
#include <vector>

template <typename T>
struct ArborX::AccessTraits<std::vector<T>>
{
  static std::size_t size(std::vector<T> const &v) { return v.size(); }
  static T const &get(std::vector<T> const &v, std::size_t i) { return v[i]; }
  using memory_space = Kokkos::HostSpace;
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using Point = ArborX::Point<3>;

  std::vector<Point> points;
  // Fill vector with random points in [-1, 1]^3
  std::uniform_real_distribution<float> dis{-1., 1.};
  std::default_random_engine gen;
  auto rd = [&]() { return dis(gen); };
  std::generate_n(std::back_inserter(points), 100, [&]() {
    return Point{rd(), rd(), rd()};
  });

  // Pass directly the vector of points to use the access traits defined above
  ArborX::BoundingVolumeHierarchy bvh{
      Kokkos::DefaultHostExecutionSpace{},
      ArborX::Experimental::attach_indices(points)};

  // As a supported alternative, wrap the vector in an unmanaged View
  bvh = ArborX::BoundingVolumeHierarchy{
      Kokkos::DefaultHostExecutionSpace{},
      ArborX::Experimental::attach_indices(
          Kokkos::View<Point *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>{
              points.data(), points.size()})};

  return 0;
}
