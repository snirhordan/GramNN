/**
 * @file   rascal/math/interpolator.hh
 *
 * @author Alexander Goscinski <alexander.goscinski@epfl.ch>
 *
 * @date   22 September 2019
 *
 * @brief Implementation of interpolator for functions functions of the form
 *        f:[x1,x2]->ℝ and f:[x1,x2]->ℝ^{n,m}
 *
 * Copyright  2019 Alexander Goscinski, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_RASCAL_MATH_INTERPOLATOR_HH_
#define SRC_RASCAL_MATH_INTERPOLATOR_HH_

#include "rascal/math/utils.hh"

/*
 * To allow flexibility of the interpolator for experimenting while giving
 * the possibility to optimize the interpolator for certain methods, the
 * interpolator class itself is only a skeleton to execute these
 * interpolator methods.
 *
 * An interpolator can be constructed given an error bound or a uniform
 * grid. The error is computed by creating a `test_grid` and compare the
 * function results with the results of the interpolator. If the error is
 * not satisfied the grid is refined and the procedure is repeated.
 *
 * Legend for documentation:
 * f,y         the function to be interpolated
 * Ω           the function is declared as f:[x1,x2]->Ω, where Ω can be ℝ or
 *             ℝ^{n,m}
 * intp, s     the interpolator function
 * grid        the grid for interpolation
 * test_grid   the test grid to estimate the error
 * ε           error function
 *
 * The interpolation method is contained in the InterpolationMethod, for the
 * creation of the grid and test grid contained in the UniformGridRational,
 * a search method to find the nearest grid point before a point for
 * interpolation contained in SearchMethod and a method for computing an
 * error term contained in ErrorMethod.
 *
 * The Interpolator class hierarchy is split into
 * Interpolator (abstract) -> YOUR_INTERPOLATOR_IMPLEMENTATION. An
 * interpolator implementation can be very general, it can be tailored for
 * flexibility with multiple combinations of methods or very focused to
 * allow for application specific optimization. The same goes for the
 * interpolator methods which can be used interchangeably with other methods
 * or can depend on specific combinations e.g. the
 * `CubicSplineScalarUniform` only works together with the
 * `UniformGridRational` class.
 *
 * While testing different methods it has turned out that the interpolator
 * is most optimal for uniform grids, therefore the interpolation methods
 * only work with uniform grids and the only supported grid type is uniform.
 *
 * There has been an implementation for an adaptive refinement method and
 * search methods for non adaptive grids, which were removed but can be
 * found here if needed:
 * https://gist.github.com/agoscinski/59b103998c7faae979dbbc672e8048f6
 */

namespace rascal {
  namespace math {
    bool is_grid_uniform(const Vector_Ref & grid);

    enum class RefinementMethod_t { Exponential, Linear };

    /**
     * UniformGridRational creates the grid and test grid. The test grid is used
     * to estimate the error of the interpolator on the grid. If the error is
     * too large, the grid is further refined until the desired accuracy is
     * achieved. UniformGridRational consists of an refinement part which
     * determines the method of refining the grid.
     *
     */
    template <RefinementMethod_t Method>
    struct UniformGridRational {
      /**
       * Computes a grid for interpolation.
       *
       * @param x1 begin of range
       * @param x2 end of range
       * @param degree_of_fineness determines the number of grid points, has to
       *        be greater than 1.
       */
      Vector_t compute_grid(double x1, double x2, int degree_of_fineness);

      /**
       * Computes a test grid to estimate the error on the `grid`. The test grid
       * is always the points between the grid points
       *
       * @param x1 begin of range
       * @param x2 end of range
       * @param degree_of_fineness determines the number of grid points, has to
       *        be greater than 1.
       */
      Vector_t compute_test_grid(double x1, double x2, int degree_of_fineness);
    };

    /**
     * Linear UniformGridRational increases the number of points by `slope`
     * times `degree_of_fineness` in each step. So the number of grid points
     * increases linearly with the `slope` parameter.
     *
     * The uniform grid uses the points in between grid points as test grid. The
     * refined grid adds `slope` number of points to grid.
     *
     *                x1                                      x2
     * grid           [       |       |       |       |       ]
     * test grid          x       x       x       x       x
     * refined grid   [    |    |    |    |    |    |    |    ] for slope = 3
     */
    template <>
    class UniformGridRational<RefinementMethod_t::Linear> {
     public:
      explicit UniformGridRational<RefinementMethod_t::Linear>(int slope = 10)
          : slope_{slope} {}

      Vector_t compute_grid(double x1, double x2,
                            int degree_of_fineness) const {
        double nb_grid_points = degree_of_fineness * slope_ + 2;
        return Vector_t::LinSpaced(nb_grid_points, x1, x2);
      }

      Vector_t compute_test_grid(double x1, double x2,
                                 int degree_of_fineness) const {
        double nb_grid_points = degree_of_fineness * slope_ + 2;
        // half of a step size in the grid to get the inner grid points
        // step size = (x2-x1)/(nb_grid_points-1)
        double offset{(x2 - x1) / (2 * (nb_grid_points - 1))};
        return Vector_t::LinSpaced(nb_grid_points - 1, x1 + offset,
                                   x2 - offset);
      }

     private:
      int slope_;
    };

    /**
     * Exponential UniformGridRational increases the number of points by
     * pow(`base`,`degree_of_fineness`) in each step.
     *
     *              x1                  x2
     * grid         [   |   |   |   |   ]
     * test grid      x   x   x   x   x
     * refined grid [ | | | | | | | | | ]  for base = 2
     */
    template <>
    class UniformGridRational<RefinementMethod_t::Exponential> {
     public:
      UniformGridRational<RefinementMethod_t::Exponential>() {}

      Vector_t compute_grid(double x1, double x2,
                            int degree_of_fineness) const {
        double nb_grid_points = 2 << degree_of_fineness;
        return Vector_t::LinSpaced(nb_grid_points, x1, x2);
      }

      Vector_t compute_test_grid(double x1, double x2,
                                 int degree_of_fineness) const {
        double nb_grid_points = 2 << degree_of_fineness;
        // half of a step size in the grid to get the inner grid points
        // step size = (x2-x1)/(nb_grid_points-1)
        double offset{(x2 - x1) / (2 * (nb_grid_points - 1))};
        return Vector_t::LinSpaced(nb_grid_points - 1, x1 + offset,
                                   x2 - offset);
      }
    };

    enum class InterpolationMethod_t {
      CubicSplineScalarUniform,
      CubicSplineVectorUniform
    };

    /**
     * Interpolation method for a function of the form f:[x1,x2]->Ω
     */
    template <InterpolationMethod_t Type>
    struct InterpolationMethod {
      /**
       * Initializes or reinitializes the interpolation method for a new grid
       * for a functions of the form f:[x1,x2]->ℝ
       *
       * @param grid the grid for interpolation
       * @param evaluated_grid evaluation of the function f(grid)
       */
      void initialize(const Vector_Ref & grid,
                      const Vector_Ref & evaluated_grid);

      /**
       * Initializes or reinitializes the interpolation method for a new grid
       * for a function of the form f:[x1,x2]->ℝ^{n,m}
       *
       * @param grid the grid for interpolation
       * @param evaluated_grid evaluation of the function f(grid) of shape
       *                       (grid_size, n*m)
       */
      void initialize(const Vector_Ref & grid,
                      const Matrix_Ref & evaluated_grid);

      /**
       * intp(x) for functions of the form f:[x1,x2]->ℝ.
       *
       * @param grid the grid for interpolation
       * @param evaluated_grid evaluation of the function f(grid)
       * @param x point to be interpolated
       * @param nearest_grid_index_to_x the nearest point to x is always
       * truncated with floor
       */
      inline double interpolate(const Vector_Ref & grid,
                                const Vector_Ref & evaluated_grid, double x,
                                int nearest_grid_index_to_x);

      /**
       * intp(x) for functions of the form f:[x1,x2]->ℝ^{n,m}.
       *
       * @param grid the grid for interpolation
       * @param evaluated_grid evaluation of the function f(grid) of shape
       *                       (grid_size, n*m)
       * @param x point to be interpolated
       * @param nearest_grid_index_to_x the nearest point to x is always
       * truncated with floor
       */
      inline double interpolate(const Vector_Ref & grid,
                                const Matrix_Ref & evaluated_grid, double x,
                                int nearest_grid_index_to_x);
    };

    /**
     * An implementation of the cubic spline method, the implementation is
     * adapted from "Numerical Recipes" [1] for functions of the form
     * f:[x1,x2]->ℝ and optimized for uniform grids, works only for uniform
     * grids. The private functions are kept as close as possible to the
     * reference.
     *
     * [1] Press, William H., et al. Numerical recipes 3rd edition: The art of
     * scientific computing. Cambridge university press, 2007.
     */
    template <>
    class InterpolationMethod<InterpolationMethod_t::CubicSplineScalarUniform> {
     public:
      InterpolationMethod() {}

      void initialize(const Vector_Ref & grid,
                      const Vector_Ref & evaluated_grid) {
        this->h = grid(1) - grid(0);
        if (this->h < DBL_FTOL) {
          throw std::runtime_error("Grid cell/step size is too small. The "
                                   "mininum supported cell/step size is " +
                                   std::to_string(DBL_FTOL) + ".");
        }
        this->h_sq_6 = this->h * this->h / 6.0;
        this->compute_second_derivative(evaluated_grid);
        this->initialized = true;
      }

      void initialize(const Vector_Ref & grid,
                      const Vector_Ref & evaluated_grid, double dfx1,
                      double dfx2) {
        this->h = grid(1) - grid(0);
        this->h_sq_6 = this->h * this->h / 6.0;
        this->compute_second_derivative(evaluated_grid, dfx1, dfx2);
        this->initialized = true;
      }

      /**
       * @pre interpolation method is initialized
       */
      double interpolate(const Vector_Ref & grid,
                         const Vector_Ref & evaluated_grid, double x,
                         int nearest_grid_index_to_x) const {
        assert(this->initialized);
        return this->interpolate_for_one_point(grid, evaluated_grid,
                                               nearest_grid_index_to_x, x);
      }

      /**
       * @pre interpolation method is initialized
       */
      double interpolate_derivative(const Vector_Ref & grid,
                                    const Vector_Ref & evaluated_grid, double x,
                                    int nearest_grid_index_to_x) const {
        assert(this->initialized);
        return this->interpolate_derivative_for_one_point(
            grid, evaluated_grid, nearest_grid_index_to_x, x);
      }

     private:
      /**
       * The two functions below implement the tridiagonal matrix algorithm
       * to solve the linear system in
       * http://mathworld.wolfram.com/CubicSpline.html
       * under different boundary conditions. We describe with s the spline
       * function and with y the targeted function for interpolation.
       *
       * Reference:
       * https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

       * clamped boundary conditions when first derivative for boundaries is
       * known s'(x_0) = f'(x_0), s'(x_n) = f'(x_n)
       */
      void compute_second_derivative(const Vector_Ref & yv, double dfx1,
                                     double dfx2);

      /**
       * Computes the second derivative for natural/simple boundary conditions
       * s''(x_0) = s''(x_n) = 0
       */
      void compute_second_derivative(const Vector_Ref & yv);

      double interpolate_for_one_point(const Vector_Ref & xx,
                                       const Vector_Ref & yy, int j1,
                                       double x) const;

      double interpolate_derivative_for_one_point(const Vector_Ref & xx,
                                                  const Vector_Ref & yy, int j1,
                                                  double x) const;

      bool initialized{false};
      // grid step size
      double h{0};
      // h*h/6.0
      double h_sq_6{0};
      /**
       * The second derivative is the solution vector of the linear system of
       * the tridiagonal toeplitz matrix derived by the conditions of cubic
       * spline. The linear system can be found in
       * http://mathworld.wolfram.com/CubicSpline.html
       *
       * In
       * addition we already multiply them with precomputed coefficients derived
       * by assuming a uniform grid second_derivative *h*h/6
       */
      Vector_t second_derivative_h_sq_6{};
    };

    /**
     * An implementation of the cubic spline method, the implementation is
     * adapted from "Numerical Recipes" [1] for functions of the form
     * y:[x1,x2]->ℝ^n and optimized for uniform grids. The private functions are
     * kept as close as possible to the reference.
     *
     * [1] Press, William H., et al. Numerical recipes 3rd edition: The art of
     * scientific computing. Cambridge university press, 2007.
     */
    template <>
    class InterpolationMethod<InterpolationMethod_t::CubicSplineVectorUniform> {
     public:
      InterpolationMethod() {}

      void initialize(const Vector_Ref & grid,
                      const Matrix_Ref & evaluated_grid) {
        this->h = grid(1) - grid(0);
        this->h_sq_6 = this->h * this->h / 6.0;
        this->compute_second_derivative(evaluated_grid);
        this->initialized = true;
      }

      /**
       * @pre interpolation method is initialized
       */
      Vector_t interpolate(const Vector_Ref & grid,
                           const Matrix_Ref & evaluated_grid, double x,
                           int nearest_grid_index_to_x) const {
        assert(this->initialized);
        return this->interpolate_for_one_point(grid, evaluated_grid,
                                               nearest_grid_index_to_x, x);
      }

      /**
       * @pre interpolation method is initialized
       */
      Vector_t interpolate_derivative(const Vector_Ref & grid,
                                      const Matrix_Ref & evaluated_grid,
                                      double x,
                                      int nearest_grid_index_to_x) const {
        assert(this->initialized);
        return this->interpolate_derivative_for_one_point(
            grid, evaluated_grid, nearest_grid_index_to_x, x);
      }

     private:
      void compute_second_derivative(const Matrix_Ref & yv);

      Vector_t interpolate_for_one_point(const Vector_Ref & xx,
                                         const Matrix_Ref & yy, int j1,
                                         double x) const;

      Vector_t interpolate_derivative_for_one_point(const Vector_Ref & xx,
                                                    const Matrix_Ref & yy,
                                                    int j1, double x) const;

      bool initialized{false};
      // The gap between two grid points
      double h{0};
      // h*h/6.0
      double h_sq_6{0};
      // second_derivative*h*h/6
      Matrix_t second_derivative_h_sq_6{};
    };

    enum class ErrorMetric_t { Absolute, Relative, AbsoluteRelative };

    /**
     * To validate the error bound of the interpolator for different error
     * methods.
     */
    template <ErrorMetric_t Metric>
    struct ErrorMethod {
      /**
       * Determines if the error is below the error bound.
       *
       * @param values intp(grid) evaluation of grid by interpolator
       * @param references f(grid) evaluation of grid by the given function
       * @param error_bound the error bound to be reached
       */
      template <class Eigen_Ref>
      static bool is_error_below_bound(const Eigen_Ref & values,
                                       const Eigen_Ref & references,
                                       double error_bound);
    };

    /**
     * Determines if the relative error is below the error bound.
     *
     * Works well for interpolators with functions evaluating large values
     * values outside the range [-1,1], so for functions of the form
     * f:[x1,x2]->[a,b] a << -1 and b >> 1.
     */
    template <>
    struct ErrorMethod<ErrorMetric_t::Relative> {
      template <class Eigen_Ref>
      static bool is_error_below_bound(const Eigen_Ref & values,
                                       const Eigen_Ref & references,
                                       double error_bound) {
        return ErrorMethod<ErrorMetric_t::Relative>::compute_global_error(
                   values, references) < error_bound;
      }
      /**
       * Computes error over the whole grid.
       */
      template <class Eigen_Ref>
      static double compute_global_error(const Eigen_Ref & values,
                                         const Eigen_Ref & references) {
        auto error{
            ErrorMethod<ErrorMetric_t::Relative>::compute_pointwise_error(
                values, references)
                .colwise()
                .mean()
                .maxCoeff()};
        return error;
      }

      /**
       * Computes error over the whole grid per grid point.
       */
      static Vector_t compute_pointwise_error(const Vector_Ref & values,
                                              const Vector_Ref & references) {
        return 2 * ((values - references).array().abs() /
                    (std::numeric_limits<double>::min() + values.array().abs() +
                     references.array().abs()));
      }

      /**
       * Computes error over the whole grid per grid point for a matrix of the
       * form (grid_size, n*m)
       */
      static Matrix_t compute_pointwise_error(const Matrix_Ref & values,
                                              const Matrix_Ref & references) {
        return 2 * ((values - references).array().abs() /
                    (std::numeric_limits<double>::min() + values.array().abs() +
                     references.array().abs()));
      }
    };

    /**
     * Determines if the absolute error is below the error bound.
     *
     * Works well for interpolators with functions evaluating small absolute
     * values inside the range [-1,1], so for functions of the form
     * f:[x1,x2]->[-1,1].
     */
    template <>
    struct ErrorMethod<ErrorMetric_t::Absolute> {
      template <class Eigen_Ref>
      static bool is_error_below_bound(const Eigen_Ref & values,
                                       const Eigen_Ref & references,
                                       double error_bound) {
        return ErrorMethod<ErrorMetric_t::Absolute>::compute_global_error(
                   values, references) < error_bound;
      }

      /**
       * Computes error over the whole grid per grid point for a matrix of the
       * form (grid_size, n*m)
       */
      template <class Eigen_Ref>
      static double compute_global_error(const Eigen_Ref & values,
                                         const Eigen_Ref & references) {
        return ErrorMethod<ErrorMetric_t::Absolute>::compute_pointwise_error(
                   values, references)
            .mean();
      }
      /**
       * Computes error over the whole grid per grid point.
       */
      static Vector_t compute_pointwise_error(const Vector_Ref & values,
                                              const Vector_Ref & references) {
        return (values - references).array().abs();
      }

      /**
       * Computes error over the whole grid per grid point for a matrix of the
       * form (grid_size, n*m)
       */
      static Matrix_t compute_pointwise_error(const Matrix_Ref & values,
                                              const Matrix_Ref & references) {
        return (values - references).array().abs();
      }
    };

    /**
     * Determines if the relative or the absolute error is below the error
     * bound.
     */
    template <>
    struct ErrorMethod<ErrorMetric_t::AbsoluteRelative> {
      template <class Eigen_Ref>
      static bool is_error_below_bound(const Eigen_Ref & values,
                                       const Eigen_Ref & references,
                                       double error_bound) {
        double absolute_error{
            ErrorMethod<ErrorMetric_t::Absolute>::compute_global_error(
                values, references)};
        double relative_error{
            ErrorMethod<ErrorMetric_t::Relative>::compute_global_error(
                values, references)};
        return (absolute_error < error_bound || relative_error < error_bound);
      }
    };

    /**
     * The purpose of the search method is to obtain from a point x in
     * the range [x1,x2] the nearest grid point. The nearest the grid point
     * via the floor rounding of x.
     *
     *                     x1     x        x2
     *                            |
     * grid                [   |   |   |   ]
     * nearest grid point      |
     */
    class UniformSearchMethod {
     public:
      /**
       * @param grid used for searching the index
       * @param nb_support_points support points of the interpolation method
       */
      void initialize(const Vector_Ref & grid, size_t nb_support_points);

      /**
       * Returns the nearest grid point.
       *
       * @param grid used for searching the index
       * @param x as mentioned in the class description
       * @return nearest grid point
       */
      int search(double x, const Vector_Ref &) const;

     private:
      // the number of support methods the interpolation method uses
      double nb_grid_points_per_unit{0};
      double x1{0};
      // asserts that we only use it with CubicSpline
      size_t nb_support_points{2};
      size_t grid_size{0};
      int search_size{0};
    };

    /**
     * Templated interpolator class, used as basic class for different kind of
     * interpolators (currently scalar, vector) to reduce code repetition and
     * allow additional optimization for specific types of interpolators.
     */
    template <class InterpolationMethod, class UniformGridRational,
              class ErrorMethod>
    class Interpolator {
     public:
      /**
       * Different constructors for the interpolator to be able to run the
       * interpolator with only the required parameters, but also to fine tune
       * different functionalities for an expert use. The detailed explanation
       * of each parameter can be seen in the protected constructor. Sets the
       * interpolator interpolating a function of the form y:[x1,x2]->Ω up to an
       * error bound or a max number of grid points in the range [x1,x2], where
       * Ω depends on the child inheriting from this class. Currently, there are
       * implementations for Ω equal ℝ or ℝ^{n,m}.
       *
       * @param function the function to be interpolated referred as y here
       * @param x1 begin range
       * @param x2 end range
       * @param error_bound the minimal error bound fulfilled
       * @param max_grid_points maximal number of grid points for the
       *        interpolating grid until the refinement of the grid stops.
       * @param initial_degree_of_fineness starting value for the
       *        `degree_of_fineness` parameter. The `degree_of_fineness`
       *        parameter describes the fineness of the grid for grid rationals
       *        with non adaptive refinement methods. Higher values results in
       *        finer grids. It depends on the specifics of the
       *        UniformGridRational how the fineness changes the number of grid
       *        points exactly. The fineness parameter should be at least 1.
       *        Starting with a very low parameter can lead to poor estimations
       *        of the errors, because the fineness of the test grid also relies
       *        on this parameter A good inital fineness value can reduce the
       *        number of steps in the initialization.
       *
       * @throw std::logic_error range is illdefined
       * @throw std::logic_error error bound is illdefined
       * @throw std::logic_error fineness is illdefined.
       */
      Interpolator(double x1, double x2, double error_bound,
                   int max_grid_points, int initial_degree_of_fineness)
          : Interpolator(x1, x2, error_bound, max_grid_points,
                         initial_degree_of_fineness, true) {}
      Interpolator(double x1, double x2, double error_bound)
          : Interpolator(x1, x2, error_bound, 10000000, 5, true) {}

      /**
       * Constructor to initialize interpolator from grid.
       * @param grid in the range [x1,x2]
       */
      explicit Interpolator(Vector_t grid) : Interpolator(grid, true) {}
      virtual ~Interpolator() = default;

      int get_degree_of_fineness() { return this->degree_of_fineness; }

      int get_grid_size() { return this->grid.size(); }

      Vector_Ref get_grid_ref() { return Vector_Ref(this->grid); }

      Vector_t get_test_grid() {
        return this->grid_rational.compute_test_grid(this->x1, this->x2,
                                                     this->degree_of_fineness);
      }

      // virtual Ω interpolate(double x);
      // virtual Ω interpolate_derivative(double x);
      double x1{0.};
      double x2{0.};

     protected:
      /*
       * @param empty bool parameter to prevent overloading with the public
       *        constructor
       */
      Interpolator(double x1, double x2, double error_bound,
                   int max_grid_points, int initial_degree_of_fineness, bool)
          : x1{x1}, x2{x2}, error_bound{error_bound},
            max_grid_points{max_grid_points},
            degree_of_fineness{initial_degree_of_fineness},
            intp_method{InterpolationMethod()},
            grid_rational{UniformGridRational()}, search_method{
                                                      UniformSearchMethod()} {
        if (x2 < x1) {
          throw std::logic_error("x2 must be greater x1");
        }
        if (error_bound <= 0) {
          throw std::logic_error("Error bound must be > 0");
        }
        if (initial_degree_of_fineness < 1) {
          throw std::logic_error("Starting grid fineness must be at least 1.");
        }
        if (max_grid_points < 2) {
          throw std::logic_error(
              "Maximal number of grid points must be at least 2.");
        }
      };

      Interpolator(Vector_t grid, bool /*dummy_for_overloading*/)
          : grid{std::move(grid)} {}

      /**
       * The general procedure of the interpolatorar is to initialize the
       * ressources for the interpolation method on a grid and test if the error
       * is within a bound. If this is not the case the interpolator will refine
       * the grid and repeat its procedure.
       */
      void initialize_iteratively() {
        this->compute_grid_error();
        while (not(this->error_below_bound) &&
               this->grid.size() < this->max_grid_points) {
          this->degree_of_fineness++;
          this->compute_grid_error();
        }
      }

      virtual void compute_grid_error() = 0;

      // The boundary points of the range of interpolation
      // grid in the range [x1,x2]
      Vector_t grid{};
      double error_bound{1e-5};
      double grid_error{0.};

      int max_grid_points{10'000'000};
      int degree_of_fineness{5};
      bool error_below_bound{false};

      InterpolationMethod intp_method{};
      UniformGridRational grid_rational{};
      UniformSearchMethod search_method{};
    };

    /**
     * Specialization of the interpolator for uniform grids with cubic spline.
     * The spline methods can be also adapted for general grids, however for
     * uniform grids the computation can be much more optimized.
     */
    template <RefinementMethod_t RefinementMethod,
              class ErrorMethod_t = ErrorMethod<ErrorMetric_t::Absolute>>
    class InterpolatorScalarUniformCubicSpline
        : public Interpolator<
              InterpolationMethod<
                  InterpolationMethod_t::CubicSplineScalarUniform>,
              UniformGridRational<RefinementMethod>, ErrorMethod_t> {
     public:
      using Parent = Interpolator<
          InterpolationMethod<InterpolationMethod_t::CubicSplineScalarUniform>,
          UniformGridRational<RefinementMethod>, ErrorMethod_t>;

      /**
       * Constructor using a the function of the form f:[x1,x2]->ℝ to initialize
       * a grid by iteratively refining it until an error bound (logical) or a
       * max number of points for the grid is reached.
       *
       * @param function the function to be interpolated referred as y here
       * @param x1 begin range
       * @param x2 end range
       * @param error_bound the minimal error bound fulfilled
       * @param max_grid_points maximal number of grid points for the
       *        interpolating grid until the refinement of the grid stops.
       * @param initial_degree_of_fineness starting fineness of the grid when
       *        starting the algorithm, the fineness parameter describes the
       *        fineness of the grid for grid rationals with non adaptive
       *        refinement methods. Higher value results in finer grids. It
       *        depends on the specifics of the UniformGridRational how the
       *        fineness changes the number of grid points exactly. For adaptive
       *        refinement methods the fineness is ignored. The fineness
       *        parameter should be at least 1. A good fineness value can reduce
       *        the number of steps in the initialization.
       * @param clamped_boundary_conditions Parameter referring to the boundary
       *        condition of cubic spline. By default the cubic spline uses the
       *        natural boundary conditions, but if f'(x1) and f'(x2) are known,
       *        the clamped boundary conditions can be turned on, which can
       *        increase the accuracy of the interpolation.
       * @param dfx1 referring to f'(x1)
       * @param dfx2 referring to f'(x2)
       */
      InterpolatorScalarUniformCubicSpline(
          std::function<double(double)> function, double x1, double x2,
          double error_bound, int max_grid_points = 100000,
          int initial_degree_of_fineness = 5,
          bool clamped_boundary_conditions = false, double dfx1 = 0,
          double dfx2 = 0)
          : Parent{x1, x2, error_bound, max_grid_points,
                   initial_degree_of_fineness},
            function{function},
            clamped_boundary_conditions{clamped_boundary_conditions},
            dfx1{dfx1}, dfx2{dfx2} {
        this->initialize_iteratively();
      }

      /**
       * Constructor using a grid to initialize the interpolator in one step
       * interpolating a function of the form y:[x1,x2]->ℝ.
       *
       * @param grid a uniform grid in the range [x1,x2] with x1 and x2 as start
       * @param evaluated_grid the grid evaluated by the function referred as
       *        f(grid)
       * @param clamped_boundary_conditions By default the cubic spline uses the
       *        natural boundary conditions, but if f'(x1) and f'(x2) are known,
       *        the clamped boundary conditions can be turned on, which can
       *        increase the accuracy of the interpolation.
       *
       * @throw std::logic_error the grid size and evaluated grid size do not
       *        match in size.
       * @throw std::logic_error the given grid is not uniform.
       *
       */
      InterpolatorScalarUniformCubicSpline(Vector_t grid,
                                           Vector_t evaluated_grid)
          : Parent(grid), evaluated_grid{evaluated_grid},
            clamped_boundary_conditions{false}, dfx1{0}, dfx2{0} {
        if (grid.size() != evaluated_grid.size()) {
          throw std::logic_error(
              "The grid and evaluated grid must match in size");
        }

        if (not(is_grid_uniform(Vector_Ref(this->grid)))) {
          throw std::logic_error("The grid has to be uniform.");
        }
        this->initialize_from_computed_grid();
      }

      InterpolatorScalarUniformCubicSpline(Vector_t grid,
                                           Vector_t evaluated_grid,
                                           bool clamped_boundary_conditions,
                                           double dfx1, double dfx2)
          : Parent(grid), evaluated_grid{evaluated_grid},
            clamped_boundary_conditions{clamped_boundary_conditions},
            dfx1{dfx1}, dfx2{dfx2} {
        if (grid.size() != evaluated_grid.size()) {
          throw std::logic_error(
              "The grid and evaluated grid must match in size");
        }
        if (not(is_grid_uniform(Vector_Ref(this->grid)))) {
          throw std::logic_error("The grid has to be uniform.");
        }
        this->initialize_from_computed_grid();
      }

      /**
       * Interpolation at point x for a function of the form f:ℝ->ℝ.
       *
       * @param x point to be interpolated within range [x1,x2]
       * @return interpolation of point x, which is intp(x)
       *
       * @pre x is in range [x1,x2]
       */
      double interpolate(double x) {
        assert(x >= this->x1 && x <= this->x2);
        int nearest_grid_index_to_x{this->search_method.search(x, this->grid)};
        return this->intp_method.interpolate(this->grid, this->evaluated_grid,
                                             x, nearest_grid_index_to_x);
      }

      Vector_t interpolate(const Vector_Ref & points) {
        Vector_t interpolated_points = Vector_t::Zero(points.size());
        for (int i{0}; i < points.size(); i++) {
          interpolated_points(i) = this->interpolate(points(i));
        }
        return interpolated_points;
      }

      /**
       * Interpolation of the derivative at point x for an function of the form
       * f:ℝ->ℝ.
       *
       * @param x point to be interpolated within range [x1,x2]
       * @return interpolation of point x, estimation of f'(x)
       *
       * @pre x is in range [x1,x2]
       */
      double interpolate_derivative(double x) {
        assert(x >= this->x1 && x <= this->x2);
        int nearest_grid_index_to_x{this->search_method.search(x, this->grid)};
        return this->intp_method.interpolate_derivative(
            this->grid, this->evaluated_grid, x, nearest_grid_index_to_x);
      }

      Vector_t interpolate_derivative(const Vector_Ref & points) {
        Vector_t interpolated_points = Vector_t::Zero(points.size());
        for (int i{0}; i < points.size(); i++) {
          interpolated_points(i) = this->interpolate_derivative(points(i));
        }
        return interpolated_points;
      }

      double eval(double x) { return this->function(x); }

      // We use evaluate when the function is used and interpolate when the
      // interpolation method is used
      Vector_t eval(const Vector_Ref & grid) {
        Vector_t evaluated_grid = Vector_t::Zero(grid.size());
        for (int i{0}; i < evaluated_grid.size(); i++) {
          evaluated_grid(i) = this->eval(grid(i));
        }
        return evaluated_grid;
      }

      Vector_Ref get_evaluated_grid_ref() {
        return Vector_Ref(this->evaluated_grid);
      }

     protected:
      void initialize_from_computed_grid() {
        if (this->clamped_boundary_conditions) {
          this->intp_method.initialize(Vector_Ref(this->grid),
                                       Vector_Ref(this->evaluated_grid),
                                       this->dfx1, this->dfx2);
        } else {
          this->intp_method.initialize(Vector_Ref(this->grid),
                                       Vector_Ref(this->evaluated_grid));
        }
        this->search_method.initialize(Vector_Ref(this->grid), 2);
      }

      /**
       * Estimates the interpolator error by computing a refined test grid and
       * compares the result of the function with the interpolator
       */
      void compute_grid_error() {
        // compute the grid
        this->grid = this->grid_rational.compute_grid(this->x1, this->x2,
                                                      this->degree_of_fineness);
        // OPT(alex) use the test grid to save computation time in evaluating
        // the grid
        this->evaluated_grid = this->eval(Vector_Ref(this->grid));
        this->initialize_from_computed_grid();

        // compute test grid
        Vector_t test_grid{this->grid_rational.compute_test_grid(
            this->x1, this->x2, this->degree_of_fineness)};
        Vector_t test_grid_interpolated{
            this->interpolate(Vector_Ref(test_grid))};
        Vector_t test_grid_evaluated{this->eval(Vector_Ref(test_grid))};

        this->error_below_bound = ErrorMethod_t::is_error_below_bound(
            Matrix_Ref(test_grid_interpolated), Matrix_Ref(test_grid_evaluated),
            this->error_bound);
      }

      /**
       * This computes parameters which are needed in the benchmarks for
       * evaluation.
       */
      void compute_parameters_for_evaluation() {
        Vector_t test_grid{this->grid_rational.compute_test_grid(
            this->x1, this->x2, this->degree_of_fineness)};
        Vector_t test_grid_interpolated{this->interpolate(test_grid)};
        Vector_t test_grid_evaluated{this->eval(test_grid)};
        Vector_t error_grid{ErrorMethod_t::compute_pointwise_error(
            Vector_Ref(test_grid_interpolated),
            Vector_Ref(test_grid_evaluated))};
        this->max_grid_error = error_grid.maxCoeff();
        this->mean_grid_error = error_grid.mean();
      }

      // f:[x1,x2]->ℝ
      std::function<double(double)> function{};
      // evaluated by the function to interpolate f(grid)
      Vector_t evaluated_grid{};
      double mean_grid_error{0.};
      double max_grid_error{0.};
      /**
       * If the first derivative of the function at the boundary points is
       * known, one can initialize the interpolator with these values to obtain
       * a higher accuracy with the interpolation method. A more descriptive
       * explanation for the boundary conditions can be found in
       * https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation#Boundary_Conditions
       */
      bool clamped_boundary_conditions{false};
      // f'(x1)
      double dfx1{0.};
      // f'(x2)
      double dfx2{0.};
    };

    template <RefinementMethod_t RefinementMethod,
              class ErrorMethod_t = ErrorMethod<ErrorMetric_t::Absolute>>
    class InterpolatorMatrixUniformCubicSpline
        : public Interpolator<
              InterpolationMethod<
                  InterpolationMethod_t::CubicSplineVectorUniform>,
              UniformGridRational<RefinementMethod>, ErrorMethod_t> {
     public:
      using Parent = Interpolator<
          InterpolationMethod<InterpolationMethod_t::CubicSplineVectorUniform>,
          UniformGridRational<RefinementMethod>, ErrorMethod_t>;

      /**
       * Constructor using a the function of the form f:[x1,x2]->ℝ^{n,m} to
       * initialize a grid by iteratively refining it until an error bound
       * (logical) or a max number of points for the grid is reached.
       *
       * @param function the function to be interpolated referred as y here
       * @param x1 begin range
       * @param x2 end range
       * @param error_bound the minimal error bound fulfilled
       * @param max_grid_points maximal number of grid points for the
       *        interpolating grid until the refinement of the grid stops.
       * @param initial_degree_of_fineness starting value for the
       *        `degree_of_fineness` parameter. The `degree_of_fineness`
       *        parameter describes the fineness of the grid for grid rationals
       *        with non adaptive refinement methods. Higher values results in
       *        finer grids. It depends on the specifics of the
       *        UniformGridRational how the fineness changes the number of grid
       *        points exactly. The fineness parameter should be at least 1.
       *        Starting with a very low parameter can lead to poor estimations
       *        of the errors, because the fineness of the test grid also relies
       *        on this parameter A good inital fineness value can reduce the
       *        number of steps in the initialization.
       * @param clamped_boundary_conditions Parameter referring to the boundary
       *        condition of cubic spline. By default the cubic spline uses the
       *        natural boundary conditions, but if f'(x1) and f'(x2) are known,
       *        the clamped boundary conditions can be turned on, which can
       *        increase the accuracy of the interpolation.
       * @param dfx1 referring to f'(x1)
       * @param dfx2 referring to f'(x2)
       */
      InterpolatorMatrixUniformCubicSpline(
          std::function<Matrix_t(double)> function, double x1, double x2,
          double error_bound, int cols, int rows, int max_grid_points = 100000,
          int initial_degree_of_fineness = 5,
          bool clamped_boundary_conditions = false, double dfx1 = 0,
          double dfx2 = 0)
          : Parent{x1, x2, error_bound, max_grid_points,
                   initial_degree_of_fineness},
            function{function}, cols{cols}, rows{rows}, matrix_size{cols *
                                                                    rows},
            clamped_boundary_conditions{clamped_boundary_conditions},
            dfx1{dfx1}, dfx2{dfx2} {
        if (clamped_boundary_conditions) {
          throw std::logic_error("InterpolatorMatrixUniformCubicSpline has "
                                 "not been implemented for "
                                 "clamped_boundary_conditions=true");
        }
        this->initialize_iteratively();
      }

      /**
       * Allows initialization of interpolator with scalar function
       * interpreting it as matrix of shape (1,1)
       */
      InterpolatorMatrixUniformCubicSpline(
          std::function<double(double)> function, double x1, double x2,
          double error_bound, int max_grid_points = 100000,
          int initial_degree_of_fineness = 5,
          bool clamped_boundary_conditions = false, double dfx1 = 0,
          double dfx2 = 0)
          : Parent{x1, x2, error_bound, max_grid_points,
                   initial_degree_of_fineness},
            cols{1}, rows{1}, matrix_size{1},
            clamped_boundary_conditions{clamped_boundary_conditions},
            dfx1{dfx1}, dfx2{dfx2} {
        if (clamped_boundary_conditions) {
          throw std::logic_error("InterpolatorMatrixUniformCubicSpline has not "
                                 "been implemented for "
                                 "clamped_boundary_conditions=true");
        }
        this->function = [function](double x) {
          Matrix_t mat(1, 1);
          mat << function(x);
          return mat;
        };
        this->initialize_iteratively();
      }

      /**
       * Constructor using a grid to initialize the interpolator in one step
       * interpolating a function of the form f:[x1,x2]->ℝ^{cols, rows}
       *
       * @param grid a uniform grid in the range [x1,x2] with x1 and x2 as
       * start
       * @param evaluated_grid the grid evaluated by the function referred as
       * f(grid) with shape (grid_size, cols*rows)
       * @param cols
       * @param rows
       *
       * @throw std::logic_error grid is not uniform
       * @throw std::logic_error grid and evaluated grid do not match by size
       * @throw std::logic_error evaluated grid and cols and rows do not match
       */
      InterpolatorMatrixUniformCubicSpline(Vector_t grid,
                                           Matrix_t evaluated_grid, int cols,
                                           int rows)
          : Parent(grid), evaluated_grid{evaluated_grid}, cols{cols},
            rows{rows}, matrix_size{cols * rows} {
        if (grid.size() != evaluated_grid.rows()) {
          throw std::logic_error(
              "The grid size and evaluated grid rows must match");
        }
        if (not(is_grid_uniform(Vector_Ref(this->grid)))) {
          throw std::logic_error("The grid has to be uniform.");
        }
        if (not(evaluated_grid.cols() == cols * rows)) {
          throw std::logic_error(
              "The evaluated grid number of cols must match cols*rows");
        }
        this->initialize_from_computed_grid();
      }

      /**
       * Interpolation at point x for a function of the form f:ℝ->ℝ^{n,m}.
       *
       * @param x point to be interpolated within range [x1,x2]
       * @return interpolation of point x, which is intp(x)
       *
       * @pre x is in range [x1,x2]
       */
      Matrix_t interpolate(double x) {
        return Eigen::Map<Matrix_t>(this->interpolate_to_vector(x).data(),
                                    this->rows, this->cols);
      }

      /**
       * Interpolation of the derivative at point x for an function of the
       * form f:ℝ->ℝ^{n,m}.
       *
       * @param x point to be interpolated within range [x1,x2]
       * @return interpolation of point x, estimation of f'(x)
       *
       * @pre x is in range [x1,x2]
       */
      Matrix_t interpolate_derivative(double x) {
        return Eigen::Map<Matrix_t>(
            this->interpolate_to_vector_derivative(x).data(), this->rows,
            this->cols);
      }

      /**
       * Interpolates the point x
       *
       * @return intp(x) in the vector shape (rows*cols)
       *
       * @pre x is in range [x1,x2]
       */
      inline Vector_t interpolate_to_vector(double x) {
        assert(x >= this->x1 && x <= this->x2);
        int nearest_grid_index_to_x{this->search_method.search(x, this->grid)};
        return this->intp_method.interpolate(this->grid, this->evaluated_grid,
                                             x, nearest_grid_index_to_x);
      }

      /**
       * Interpolates the each x in points
       *
       * @return intp(x) in the vector shape (rows*cols)
       */
      inline Matrix_t interpolate_to_vector(const Vector_Ref & points) {
        Matrix_t interpolated_points =
            Matrix_t::Zero(points.size(), this->matrix_size);
        for (int i{0}; i < points.size(); i++) {
          interpolated_points.row(i) = this->interpolate_to_vector(points(i));
        }
        return interpolated_points;
      }

      /**
       * Interpolates the derivative of point x
       * @return intp(x) in the shape (rows*cols)
       *
       * @pre x is in range [x1,x2]
       */
      inline Vector_t interpolate_to_vector_derivative(const double x) {
        assert(x >= this->x1 && x <= this->x2);
        int nearest_grid_index_to_x{this->search_method.search(x, this->grid)};
        return this->intp_method.interpolate_derivative(
            this->grid, this->evaluated_grid, x, nearest_grid_index_to_x);
      }

      /**
       * Interpolates the derivative of each x in points
       * @return intp(x) in the shape (points.size(), rows*cols)
       */
      inline Matrix_t
      interpolate_to_vector_derivative(const Vector_Ref & points) {
        Matrix_t interpolated_points =
            Matrix_t::Zero(points.size(), this->matrix_size);
        for (int i{0}; i < points.size(); i++) {
          interpolated_points.row(i) =
              this->interpolate_to_vector_derivative(points(i));
        }
        return interpolated_points;
      }

      /**
       * This function exists for benchmark purposes, to estimate the memory
       * and function call overhead.
       *
       * @param x point to be interpolated within range [x1,x2]
       * @return interpolation of point x, estimating f(x)
       *
       * @assert x is not in range [x1,x2]
       */
      Matrix_t interpolate_dummy(const double x) const {
        assert(x >= this->x1 && x <= this->x2);
        return x * Matrix_t::Ones(this->rows, this->cols);
      }

      Matrix_Ref get_evaluated_grid_ref() {
        return Matrix_Ref(this->evaluated_grid);
      }

      int get_matrix_size() { return this->matrix_size; }

      /**
       * @param x
       * @return evaluation of f on x, f(x)
       */
      Vector_t eval(const double x) {
        return Eigen::Map<Vector_t>(this->function(x).data(),
                                    this->matrix_size);
      }

      // OPT(alex) container for Matrix_t, then reshape one time to prevent
      // check overflow
      /**
       * @param grid
       * @return evaluation of f on each point of the grid, f(grid)
       */
      Matrix_t eval(const Vector_Ref & grid) {
        Matrix_t evaluated_grid =
            Matrix_t::Zero(grid.size(), this->matrix_size);
        for (int i{0}; i < grid.size(); i++) {
          evaluated_grid.row(i) = this->eval(grid(i));
        }
        return evaluated_grid;
      }

     protected:
      // ensures that the grid for estimating the test error is fine enough.
      void compute_grid_error() {
        this->grid = this->grid_rational.compute_grid(this->x1, this->x2,
                                                      this->degree_of_fineness);
        this->evaluated_grid = this->eval(Vector_Ref(this->grid));
        this->initialize_from_computed_grid();

        // ensures that the grid for estimating the test error is fine enough.
        int test_degree_of_fineness{std::max(this->degree_of_fineness, 5)};
        Vector_t test_grid{this->grid_rational.compute_test_grid(
            this->x1, this->x2, test_degree_of_fineness)};
        // (grid_size, row*col)
        // OPT(alex) check how this is optimized
        Matrix_t test_grid_interpolated{
            this->interpolate_to_vector(Vector_Ref(test_grid))};
        Matrix_t test_grid_evaluated{this->eval(Vector_Ref(test_grid))};
        this->error_below_bound = ErrorMethod_t::is_error_below_bound(
            Matrix_Ref(test_grid_interpolated), Matrix_Ref(test_grid_evaluated),
            this->error_bound);
      }

      void initialize_from_computed_grid() {
        if (this->clamped_boundary_conditions) {
          // OPT(alex) implement for clamped_boundary_conditions
          throw std::logic_error("Clamped boundary condition has yet not been "
                                 "implemented for CubicSplineVectorUniform.");
        } else {
          this->intp_method.initialize(Vector_Ref(this->grid),
                                       Matrix_Ref(this->evaluated_grid));
        }
        this->search_method.initialize(Vector_Ref(this->grid), 2);
      }

      // f:[x1,x2]->ℝ^{rows,cols}
      std::function<Matrix_t(double)> function{};
      // f(grid) reshaped to (grid_size, n*m)
      Matrix_t evaluated_grid{};
      int cols{0};
      int rows{0};
      // cols*rows
      int matrix_size{0};
      /**
       * If the first derivative of the function at the boundary points is
       * known, one can initialize the interpolator with these values to
       * obtain a higher accuracy with the interpolation method. A more
       * descriptive explanation for the boundary conditions can be found in
       * https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation#Boundary_Conditions
       */
      bool clamped_boundary_conditions{false};
      // f'(x1)
      double dfx1{0.};
      // f'(x2)
      double dfx2{0.};
    };

  }  // namespace math
}  // namespace rascal

#endif  // SRC_RASCAL_MATH_INTERPOLATOR_HH_
