/**
 * @file   gauss_legendre.cc
 *
 * @author  Felix Musil <felix.musil@epfl.ch>
 * @author  Max Veit <max.veit@epfl.ch>
 *
 * @date   27 May 2019
 *
 * @brief Implementation of the gauss legendre quadrature weights and points
 *
 * Copyright  2019  Felix Musil, Max Veit COSMO (EPFL), LAMMM (EPFL)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "rascal/math/gauss_legendre.hh"
using namespace rascal::math;  // NOLINT

MatrixX2_t rascal::math::compute_gauss_legendre_points_weights(double r_st,
                                                               double r_nd,
                                                               int order_n) {
  if (order_n < 2) {
    throw std::runtime_error(R"(there should be at least 2 integration
        points but: order_n < 2 == true)");
  }
  MatrixX2_t point_weight(order_n, 2);

  double z{0.}, pp{0.}, r_midpoint{(r_st + r_nd) * 0.5},
      r_length{0.5 * (r_nd - r_st)};

  int m{static_cast<int>((order_n + 1) / 2)};
  // the weights and points are symmetric so only half need to be computed
  for (int ii{0}; ii < m; ++ii) {
    // starting guess for the ith root
    z = std::cos(math::PI * (ii + 0.75) / (order_n + 0.5));
    double z1{0.};
    while (std::abs(z - z1) > DBL_FTOL) {
      double p1{1.}, p2{0.};
      // use LP 3-term recurrence relation with p1/2/3
      for (int jj{0}; jj < order_n; ++jj) {
        double p3{p2};
        p2 = p1;
        p1 = ((2.0 * jj + 1.0) * z * p2 - jj * p3) / (jj + 1);
      }
      // compute the derivative of the LP
      pp = order_n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      // Newton's method
      z = z1 - p1 / pp;
    }

    // scale the root to the [r_st, r_nd] interval
    point_weight(ii, 0) = r_midpoint - r_length * z;
    point_weight(order_n - 1 - ii, 0) = r_midpoint + r_length * z;
    // compute the weight
    point_weight(ii, 1) = (2.0 * r_length) / ((1.0 - z * z) * pp * pp);
    point_weight(order_n - 1 - ii, 1) = point_weight(ii, 1);
  }
  return point_weight;
}
