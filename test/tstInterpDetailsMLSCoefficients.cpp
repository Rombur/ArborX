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

#include "ArborX_EnableDeviceTypes.hpp"
#include "ArborX_EnableViewComparison.hpp"
#include <detail/ArborX_InterpDetailsCompactRadialBasisFunction.hpp>
#include <detail/ArborX_InterpDetailsMovingLeastSquaresCoefficients.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <type_traits>

template <typename ExecutionSpace, typename SourceValues, typename Coefficients>
Kokkos::View<double *, typename SourceValues::memory_space>
interpolate(ExecutionSpace const &space, SourceValues const &source_values,
            Coefficients const &coeffs)
{
  int num_targets = coeffs.extent(0);
  int num_neighbors = coeffs.extent(1);

  Kokkos::View<double *, typename SourceValues::memory_space> target_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::target_values"),
      num_targets);
  Kokkos::parallel_for(
      "Testing::mls_coefficients::target_interpolation",
      Kokkos::RangePolicy(space, 0, num_targets), KOKKOS_LAMBDA(int const i) {
        double tmp = 0;
        for (int j = 0; j < num_neighbors; j++)
          tmp += coeffs(i, j) * source_values(i, j);
        target_values(i) = tmp;
      });

  return target_values;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mls_coefficients, DeviceType, ARBORX_DEVICE_TYPES)
{
  // FIXME_HIP: the CI fails with:
  // fatal error: in "mls_coefficients_edge_cases<Kokkos__Device<Kokkos__HIP_
  // Kokkos__HIPSpace>>": std::runtime_error: Kokkos::Impl::ParallelFor/Reduce<
  // HIP > could not find a valid team size.
  // The error seems similar to https://github.com/kokkos/kokkos/issues/6743
#ifdef KOKKOS_ENABLE_HIP
  if (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
  {
    return;
  }
#endif
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  // Case 1: f(x) = 3, 2 neighbors, linear
  //      -------0--------------->
  // SRC:        0   2   4   6
  // TGT:          1   3   5
  using Point0 = ArborX::Point<1, double>;
  Kokkos::View<Point0 **, MemorySpace> srcp0("Testing::srcp0", 3, 2);
  Kokkos::View<Point0 *, MemorySpace> tgtp0("Testing::tgtp0", 3);
  Kokkos::View<double **, MemorySpace> srcv0("Testing::srcv0", 3, 2);
  Kokkos::View<double *, MemorySpace> tgtv0("Testing::tgtv0", 3);
  Kokkos::parallel_for(
      "Testing::mls_coefficients::for0", Kokkos::RangePolicy(space, 0, 3),
      KOKKOS_LAMBDA(int const i) {
        srcp0(i, 0) = {{2. * i}};
        srcp0(i, 1) = {{2. * i + 2}};
        tgtp0(i) = {{2. * i + 1}};

        auto f = [](const Point0 &) { return 3.; };

        srcv0(i, 0) = f(srcp0(i, 0));
        srcv0(i, 1) = f(srcp0(i, 1));
        tgtv0(i) = f(tgtp0(i));
      });
  auto coeffs0 = ArborX::Interpolation::Details::movingLeastSquaresCoefficients<
      ArborX::Interpolation::CRBF::Wendland<0>,
      ArborX::Interpolation::PolynomialDegree<1>, double>(space, srcp0, tgtp0);
  auto eval0 = interpolate(space, srcv0, coeffs0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: f(x, y) = xy + 4x, 8 neighbors, quad
  //        ^
  //        |
  //    S   S   S
  //      T | T
  // ---S---S---S--->
  //      T | T
  //    S   S   S
  //        |
  using Point1 = ArborX::Point<2, double>;
  Kokkos::View<Point1 **, MemorySpace> srcp1("Testing::srcp1", 4, 8);
  Kokkos::View<Point1 *, MemorySpace> tgtp1("Testing::tgtp1", 4);
  Kokkos::View<double **, MemorySpace> srcv1("Testing::srcv1", 4, 8);
  Kokkos::View<double *, MemorySpace> tgtv1("Testing::tgtv1", 4);
  Kokkos::parallel_for(
      "Testing::mls_coefficients::for1", Kokkos::RangePolicy(space, 0, 4),
      KOKKOS_LAMBDA(int const i) {
        int u = (i / 2) * 2 - 1;
        int v = (i % 2) * 2 - 1;
        for (int j = 0, k = 0; j < 9; j++)
        {
          int x = (j / 3) - 1;
          int y = (j % 3) - 1;
          if (x == -u && y == -v)
            continue;

          srcp1(i, k) = {{x * 2., y * 2.}};
          k++;
        }
        tgtp1(i) = {{double(u), double(v)}};

        auto f = [](const Point1 &p) { return p[0] * p[1] + 4 * p[0]; };

        for (int j = 0; j < 8; j++)
          srcv1(i, j) = f(srcp1(i, j));
        tgtv1(i) = f(tgtp1(i));
      });
  auto coeffs1 = ArborX::Interpolation::Details::movingLeastSquaresCoefficients<
      ArborX::Interpolation::CRBF::Wendland<2>,
      ArborX::Interpolation::PolynomialDegree<2>, double>(space, srcp1, tgtp1);
  auto eval1 = interpolate(space, srcv1, coeffs1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mls_coefficients_edge_cases, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // FIXME_HIP: the CI fails with:
  // fatal error: in "mls_coefficients_edge_cases<Kokkos__Device<Kokkos__HIP_
  // Kokkos__HIPSpace>>": std::runtime_error: Kokkos::Impl::ParallelFor/Reduce<
  // HIP > could not find a valid team size.
  // The error seems similar to https://github.com/kokkos/kokkos/issues/6743
#ifdef KOKKOS_ENABLE_HIP
  if (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
  {
    return;
  }
#endif
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  // Case 1: Same as previous case 1, but points are 2D and locked on y=0
  using Point0 = ArborX::Point<2, double>;
  Kokkos::View<Point0 **, MemorySpace> srcp0("Testing::srcp0", 3, 2);
  Kokkos::View<Point0 *, MemorySpace> tgtp0("Testing::tgtp0", 3);
  Kokkos::View<double **, MemorySpace> srcv0("Testing::srcv0", 3, 2);
  Kokkos::View<double *, MemorySpace> tgtv0("Testing::tgtv0", 3);
  Kokkos::parallel_for(
      "Testing::mls_coefficients_edge_cases::for0",
      Kokkos::RangePolicy(space, 0, 3), KOKKOS_LAMBDA(int const i) {
        srcp0(i, 0) = {{2. * i, 0.}};
        srcp0(i, 1) = {{2. * i + 2, 0.}};
        tgtp0(i) = {{2. * i + 1, 0.}};

        auto f = [](const Point0 &) { return 3.; };

        srcv0(i, 0) = f(srcp0(i, 0));
        srcv0(i, 1) = f(srcp0(i, 1));
        tgtv0(i) = f(tgtp0(i));
      });
  auto coeffs0 = ArborX::Interpolation::Details::movingLeastSquaresCoefficients<
      ArborX::Interpolation::CRBF::Wendland<0>,
      ArborX::Interpolation::PolynomialDegree<1>, double>(space, srcp0, tgtp0);
  auto eval0 = interpolate(space, srcv0, coeffs0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: Same but corner source points are also targets
  using Point1 = ArborX::Point<2, double>;
  Kokkos::View<Point1 **, MemorySpace> srcp1("Testing::srcp1", 4, 8);
  Kokkos::View<Point1 *, MemorySpace> tgtp1("Testing::tgtp1", 4);
  Kokkos::View<double **, MemorySpace> srcv1("Testing::srcv1", 4, 8);
  Kokkos::View<double *, MemorySpace> tgtv1("Testing::tgtv1", 4);
  Kokkos::parallel_for(
      "Testing::mls_coefficients_edge_cases::for1",
      Kokkos::RangePolicy(space, 0, 4), KOKKOS_LAMBDA(int const i) {
        int u = (i / 2) * 2 - 1;
        int v = (i % 2) * 2 - 1;
        for (int j = 0, k = 0; j < 9; j++)
        {
          int x = (j / 3) - 1;
          int y = (j % 3) - 1;
          if (x == -u && y == -v)
            continue;

          srcp1(i, k) = {{x * 2., y * 2.}};
          k++;
        }
        tgtp1(i) = {{u * 2., v * 2.}};

        auto f = [](const Point1 &p) { return p[0] * p[1] + 4 * p[0]; };

        for (int j = 0; j < 8; j++)
          srcv1(i, j) = f(srcp1(i, j));
        tgtv1(i) = f(tgtp1(i));
      });
  auto coeffs1 = ArborX::Interpolation::Details::movingLeastSquaresCoefficients<
      ArborX::Interpolation::CRBF::Wendland<2>,
      ArborX::Interpolation::PolynomialDegree<2>, double>(space, srcp1, tgtp1);
  auto eval1 = interpolate(space, srcv1, coeffs1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}
