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
#ifndef ARBORX_RAY_HPP
#define ARBORX_RAY_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // equal
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Triangle.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cmath>

namespace ArborX
{
namespace Experimental
{

struct Vector : private Point
{
  using Point::Point;
  using Point::operator[];
  friend KOKKOS_FUNCTION constexpr bool operator==(Vector const &v,
                                                   Vector const &w)
  {
    return v[0] == w[0] && v[1] == w[1] && v[2] == w[2];
  }
};

KOKKOS_INLINE_FUNCTION constexpr Vector makeVector(Point const &begin,
                                                   Point const &end)
{
  Vector v;
  for (int d = 0; d < 3; ++d)
  {
    v[d] = end[d] - begin[d];
  }
  return v;
}

KOKKOS_INLINE_FUNCTION constexpr auto dotProduct(Vector const &v,
                                                 Vector const &w)
{
  return v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
}

KOKKOS_INLINE_FUNCTION constexpr Vector crossProduct(Vector const &v,
                                                     Vector const &w)
{
  return {v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2],
          v[0] * w[1] - v[1] * w[0]};
}

KOKKOS_INLINE_FUNCTION constexpr bool equals(Vector const &v, Vector const &w)
{
  return v == w;
}

struct Ray
{
  Point _origin = {};
  Vector _direction = {};

  KOKKOS_DEFAULTED_FUNCTION
  constexpr Ray() = default;

  KOKKOS_FUNCTION
  Ray(Point const &origin, Vector const &direction)
      : _origin(origin)
      , _direction(direction)
  {
    normalize(_direction);
  }

  KOKKOS_FUNCTION
  constexpr Point &origin() { return _origin; }

  KOKKOS_FUNCTION
  constexpr Point const &origin() const { return _origin; }

  KOKKOS_FUNCTION
  constexpr Vector &direction() { return _direction; }

  KOKKOS_FUNCTION
  constexpr Vector const &direction() const { return _direction; }

private:
  // We would like to use Scalar defined as:
  // using Scalar = std::decay_t<decltype(std::declval<Vector>()[0])>;
  // However, this means using float to compute the norm. This creates a large
  // error in the norm that affects ray tracing for triangles. Casting the
  // norm from double to float once it has been computed is not enough to
  // improve the value of the normalized vector. Thus, the norm has to return a
  // double.
  using Scalar = double;

  KOKKOS_FUNCTION
  static Scalar norm(Vector const &v)
  {
    Scalar sq{};
    for (int d = 0; d < 3; ++d)
      sq += static_cast<Scalar>(v[d]) * static_cast<Scalar>(v[d]);
    return std::sqrt(sq);
  }

  KOKKOS_FUNCTION static void normalize(Vector &v)
  {
    auto const magv = norm(v);
    assert(magv > 0);
    for (int d = 0; d < 3; ++d)
      v[d] /= magv;
  }
};

KOKKOS_INLINE_FUNCTION
constexpr bool equals(Ray const &l, Ray const &r)
{
  using ArborX::Details::equals;
  return equals(l.origin(), r.origin()) && equals(l.direction(), r.direction());
}

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Ray const &ray) { return ray.origin(); }

// The ray-box intersection algorithm is based on [1]. Their 'efficient slag'
// algorithm checks the intersections both in front and behind the ray.
//
// There are few issues here. First, when a ray direction is aligned with one
// of the axis, a division by zero will occur. This is fine, as usually it
// results in +inf or -inf, which are treated correctly. However, it also leads
// to the second situation, when it is 0/0 which occurs when the ray's origin
// in that dimension is on the same plane as one of the corners of the box
// (i.e., if inv_ray_dir[d] == 0 && (min_corner[d] == origin[d] || max_corner[d]
// == origin[d])). This leads to NaN, which are not treated correctly (unless,
// as in [1], the underlying min/max functions are able to ignore them). The
// issue is discussed in more details in [2] and the website (key word: A
// minimal ray-tracer: rendering simple shapes).
//
// [1] Majercik, A., Crassin, C., Shirley, P., & McGuire, M. (2018). A ray-box
// intersection algorithm and efficient dynamic voxel rendering. Journal of
// Computer Graphics Techniques Vol, 7(3).
//
// [2] Williams, A., Barrus, S., Morley, R. K., & Shirley, P. (2005). An
// efficient and robust ray-box intersection algorithm. In ACM SIGGRAPH 2005
// Courses (pp. 9-es).
KOKKOS_INLINE_FUNCTION
bool intersection(Ray const &ray, Box const &box, float &tmin, float &tmax)
{
  auto const &min = box.minCorner();
  auto const &max = box.maxCorner();
  auto const &orig = ray.origin();
  auto const &dir = ray.direction();

  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = -inf;
  tmax = inf;

  for (int d = 0; d < 3; ++d)
  {
    float tdmin;
    float tdmax;
    if (dir[d] >= 0)
    {
      tdmin = (min[d] - orig[d]) / dir[d];
      tdmax = (max[d] - orig[d]) / dir[d];
    }
    else
    {
      tdmin = (max[d] - orig[d]) / dir[d];
      tdmax = (min[d] - orig[d]) / dir[d];
    }
    if (tmin < tdmin)
      tmin = tdmin;
    if (tmax > tdmax)
      tmax = tdmax;
  }
  return (tmin <= tmax);
}

KOKKOS_INLINE_FUNCTION
bool intersects(Ray const &ray, Box const &box)
{
  float tmin;
  float tmax;
  // intersects only if box is in front of the ray
  return intersection(ray, box, tmin, tmax) && (tmax >= 0.f);
}

// rotate to the x-axis clockwise
KOKKOS_INLINE_FUNCTION Point rotate2D(Point const &point)
{
  Point point_star;
  float r = std::sqrt(point[0] * point[0] + point[1] * point[1]);
  float costheta;
  float sintheta;
  if (r != 0.0f)
  {
    costheta = std::fabs(point[0]) / r;
    sintheta = std::fabs(point[1]) / r;
  }
  point_star[0] = point[0] * costheta + point[1] * sintheta;
  point_star[1] = point[2];
  point_star[2] = 0.0;
  return point_star;
}

KOKKOS_INLINE_FUNCTION bool rayEdgeIntersect(Point const &edge_vertex_1,
                                             Point const &edge_vertex_2,
                                             float &t)
{
  // modified from Bruno's PR
  // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
  float x3 = edge_vertex_1[0];
  float y3 = edge_vertex_1[1];
  float x4 = edge_vertex_2[0];
  float y4 = edge_vertex_2[1];

  float y1 = KokkosExt::min(y3, y4);
  float y2 = KokkosExt::max(y3, y4);

  // float det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  float det = -(y1 - y2) * (x3 - x4);
  //  auto const epsilon = 0.000001f;
  //  if (det > -epsilon && det < epsilon)
  if (det == 0.0f)
  {
    return false;
  }
  // t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det;
  t = (-x3 * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det;

  if (t >= 0.)
  {
    // float u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / det;
    float u = -x3 * (y1 - y2) / det;
    if (u >= 0. && u <= 1.)
      return true;
  }
  return false;
}

// Watertight Ray/Triangle Intersection
// [1] Woop, S. et al. (2013).
// Journal of Computer Graphics Techniques Vol. 2(1)
KOKKOS_INLINE_FUNCTION
bool intersection(Ray const &ray, Triangle const &triangle, float &tmin,
                  float &tmax)
{
  auto dir = ray.direction();
  // normalizing the direction vector by its largest component.
  // if precalculated, kx, ky, kz need to be stored.
  int kz = 0;
  for (int i = 1; i < 3; i++)
  {
    float compmax = fabs(dir[i - 1]);
    if (fabs(dir[i]) > compmax)
    {
      compmax = dir[i];
      kz = i;
    }
  }
  int kx = kz + 1;
  if (kx == 3)
    kx = 0;
  int ky = kx + 1;
  if (ky == 3)
    ky = 0;

  if (dir[kz] < 0.0f)
    KokkosExt::swap(kx, ky);

  Vector s;

  s[2] = 1.0f / dir[kz];
  s[0] = dir[kx] * s[2];
  s[1] = dir[ky] * s[2];

  // calculate vertices relative to ray origin
  Vector const oA = makeVector(ray.origin(), triangle.a);
  Vector const oB = makeVector(ray.origin(), triangle.b);
  Vector const oC = makeVector(ray.origin(), triangle.c);

  Point A, B, C;

  // perform shear and scale of vertices
  A[0] = oA[kx] - s[0] * oA[kz];
  A[1] = oA[ky] - s[1] * oA[kz];
  B[0] = oB[kx] - s[0] * oB[kz];
  B[1] = oB[ky] - s[1] * oB[kz];
  C[0] = oC[kx] - s[0] * oC[kz];
  C[1] = oC[ky] - s[1] * oC[kz];

  // calculate scaled barycentric coordinates
  float u = C[0] * B[1] - C[1] * B[0];
  float v = A[0] * C[1] - A[1] * C[0];
  float w = B[0] * A[1] - B[1] * A[0];

  // fallback to edge test using double precision
  if (u == 0.0f || v == 0.0f || w == 0.0f)
  {
    double CxBy = (double)C[0] * (double)B[1];
    double CyBx = (double)C[1] * (double)B[0];
    u = (float)(CxBy - CyBx);

    double AxCy = (double)A[0] * (double)C[1];
    double AyCx = (double)A[1] * (double)C[0];
    v = (float)(AxCy - AyCx);

    double BxAy = (double)B[0] * (double)A[1];
    double ByAx = (double)B[1] * (double)A[0];
    w = (float)(BxAy - ByAx);
  }

  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = inf;
  tmax = -inf;

  // depending on the facing of the triangle
  if ((u < 0.0f || v < 0.0f || w < 0.0f) && (u > 0.0f || v > 0.0f || w > 0.0f))
    return false;

  // calculate determinant
  float det = u + v + w;

  A[2] = s[2] * oA[kz];
  B[2] = s[2] * oB[kz];
  C[2] = s[2] * oC[kz];

  if (det != 0.0f)
  {
    float t = (u * A[2] + v * B[2] + w * C[2]) / det;
    tmax = t;
    tmin = t;
    return tmax >= tmin;
  }
  else
  {
    // the ray is co-planar to the triangle
    // check the intersection with each edge
    //
    // AB
    auto A_star = rotate2D(A);
    auto B_star = rotate2D(B);
    auto C_star = rotate2D(C);

    float t_ab = inf;
    bool ab_intersect = rayEdgeIntersect(A_star, B_star, t_ab);
    if (ab_intersect)
    {
      tmin = t_ab;
      tmax = t_ab;
    }
    float t_bc = inf;
    bool bc_intersect = rayEdgeIntersect(B_star, C_star, t_bc);
    if (bc_intersect)
    {
      tmin = KokkosExt::min(tmin, t_bc);
      tmax = KokkosExt::max(tmax, t_bc);
    }
    float t_ca = inf;
    bool ca_intersect = rayEdgeIntersect(C_star, A_star, t_ca);
    if (ca_intersect)
    {
      tmin = KokkosExt::min(tmin, t_ca);
      tmax = KokkosExt::max(tmax, t_ca);
    }

    if (ab_intersect || bc_intersect || ca_intersect)
    {
      tmin = tmin / det;
      tmax = tmax / det;
      return true;
    }
    else
    {
      return false;
    }
  }
}

KOKKOS_INLINE_FUNCTION
bool intersects(Ray const &ray, Triangle const &triangle)
{
  float tmin;
  float tmax;
  // intersects only if triangle is in front of the ray
  return intersection(ray, triangle, tmin, tmax) && (tmax >= 0.f);
}

// Solves a*x^2 + b*x + c = 0.
// If a solution exists, return true and stores roots at x1, x2.
// If a solution does not exist, returns false.
KOKKOS_INLINE_FUNCTION bool solveQuadratic(float const a, float const b,
                                           float const c, float &x1, float &x2)
{
  assert(a != 0);

  auto const discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
    return false;
  if (discriminant == 0)
  {
    x1 = x2 = -b / (2 * a);
    return true;
  }

  // Instead of doing a simple
  //    (-b +- std::sqrt(discriminant)) / (2*a)
  // we use a more stable algorithm with less loss of precision (see, for
  // clang-format off
  // example, https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection).
  // clang-format on
  auto const q = (b > 0) ? (-b - std::sqrt(discriminant)) / 2.0
                         : (-b + std::sqrt(discriminant)) / 2.0;
  x1 = q / a;
  x2 = c / q;

  return true;
}

// Ray-Sphere intersection algorithm.
//
// The sphere can be expressed as the solution to
//     |p - c|^2 - r^2 = 0,           (1)
// where c is the center of the sphere, and r is the radius. On the other
// hand, any point on a bidirectional ray satisfies
//     p = o + t*d,                   (2)
// where o is the origin, and d is the direction vector.
// Substituting (2) into (1),
//     |(o + t*d) - c|^2 - r^2 = 0,   (3)
// results in a quadratic equation for unknown t
//     a2 * t^2 + a1 * t + a0 = 0
// with
//     a2 = |d|^2, a1 = 2*(d, o - c), and a0 = |o - c|^2 - r^2.
// Then, we only need to intersect the solution interval [tmin, tmax] with
// [0, +inf) for the unidirectional ray.
KOKKOS_INLINE_FUNCTION bool intersection(Ray const &ray, Sphere const &sphere,
                                         float &tmin, float &tmax)
{
  auto const &r = sphere.radius();

  // Vector oc = (origin_of_ray - center_of_sphere)
  Vector const oc = makeVector(sphere.centroid(), ray.origin());

  float const a2 = 1.f; // directions are normalized
  float const a1 = 2.f * dotProduct(ray.direction(), oc);
  float const a0 = dotProduct(oc, oc) - r * r;

  if (solveQuadratic(a2, a1, a0, tmin, tmax))
  {
    // ensures that tmin <= tmax
    if (tmin > tmax)
      KokkosExt::swap(tmin, tmax);

    return true;
  }
  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  tmin = inf;
  tmax = -inf;
  return false;
}

KOKKOS_INLINE_FUNCTION float overlapDistance(Ray const &ray,
                                             Sphere const &sphere)
{
  float tmin;
  float tmax;
  if (!intersection(ray, sphere, tmin, tmax) || (tmax < 0))
  {
    return 0.f;
  }

  // Overlap [tmin, tmax] with [0, +inf)
  tmin = KokkosExt::max(0.f, tmin);

  // As direction is normalized,
  //   |(o + tmax*d) - (o + tmin*d)| = tmax - tmin
  return (tmax - tmin);
}

} // namespace Experimental
} // namespace ArborX
#endif
