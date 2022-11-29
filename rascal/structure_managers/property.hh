/**
 * @file   rascal/structure_managers/property.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief  Properties of atom-, pair-, triplet-, etc-related values
 *
 * @section LICENSE
 *
 * Copyright  2018 Till Junge, Felix Musil, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_HH_

#include "rascal/structure_managers/property_typed.hh"
#include "rascal/utils/basic_types.hh"
#include "rascal/utils/utils.hh"

#include <Eigen/Dense>

#include <cassert>
#include <type_traits>

namespace rascal {

  //! Forward declaration of traits to use `Property`.
  template <class Manager>
  struct StructureManager_traits;

  /**
   * Class definition of `property`. This is a container of data (specifiable),
   * which can be access with clusters directly, without the need for dealing
   * with indices.
   */
  template <typename T, size_t Order, class Manager, Dim_t NbRow = 1,
            Dim_t NbCol = 1>
  class Property : public TypedProperty<T, Order, Manager> {
    static_assert((std::is_arithmetic<T>::value ||
                   std::is_same<T, std::complex<double>>::value),
                  "can currently only handle arithmetic types");

   public:
    using Parent = TypedProperty<T, Order, Manager>;
    using Manager_t = Manager;
    using Self_t = Property<T, Order, Manager, NbRow, NbCol>;
    using Value = internal::Value<T, NbRow, NbCol>;
    static_assert(std::is_same<Value, internal::Value<T, NbRow, NbCol>>::value,
                  "type alias failed");

    using value_type = typename Value::value_type;
    using reference = typename Value::reference;
    using const_reference = typename Value::const_reference;

    static constexpr bool IsStaticallySized{(NbCol != Eigen::Dynamic) and
                                            (NbRow != Eigen::Dynamic)};

    //! Empty type for tag dispatching to differenciate between
    //! the Dynamic and Static size case
    struct DynamicSize {};
    struct StaticSize {};

    //! Default constructor
    Property() = delete;

    //! Constructor with Manager with optional ghost exclusion
    explicit Property(Manager_t & manager, std::string metadata = "no metadata",
                      const bool & exclude_ghosts = false)
        : Parent{manager, NbRow, NbCol, metadata, exclude_ghosts},
          type_id{typeid(Self_t).name()} {}

    //! Copy constructor
    Property(const Property & other) = delete;

    //! Move constructor
    Property(Property && other) = default;

    //! Destructor
    virtual ~Property() = default;

    //! Copy assignment operator
    Property & operator=(const Property & other) = delete;

    //! Move assignment operator
    Property & operator=(Property && other) = delete;

    /* ---------------------------------------------------------------------- */
    /**
     * Cast operator: takes polymorphic base class reference, and returns
     * properly casted fully typed and sized reference, or throws a runttime
     * error
     */
    static void check_compatibility(PropertyBase & other) {
      // check ``type`` compatibility
      auto type_id{typeid(Self_t).name()};
      if (other.get_type_info() != type_id) {
        std::stringstream err_str{};
        err_str << "Incompatible types: '" << other.get_type_info() << "' != '"
                << type_id << "'.";
        throw std::runtime_error(err_str.str());
      }
    }

    const std::string & get_type_info() const final { return this->type_id; }

    /* ---------------------------------------------------------------------- */
    /**
     * allows to add a value to `Property` during construction of the
     * neighbourhood.
     */
    void push_back(reference ref) {
      // use tag dispatch to use the proper definition
      // of the push_in_vector function
      this->push_back(
          ref,
          std::conditional_t<(IsStaticallySized), StaticSize, DynamicSize>{});
    }

    /* ---------------------------------------------------------------------- */
    /**
     * Function for adding Eigen-based matrix data to `property`
     */
    template <typename Derived>
    void push_back(const Eigen::DenseBase<Derived> & ref) {
      // use tag dispatch to use the proper definition
      // of the push_in_vector function
      this->push_back(
          ref,
          std::conditional_t<(IsStaticallySized), StaticSize, DynamicSize>{});
    }

    /* ---------------------------------------------------------------------- */
    /**
     * Property accessor by cluster ref
     */
    template <size_t CallerLayer>
    reference operator[](const ClusterRefKey<Order, CallerLayer> & id) {
      // You are trying to access a property that does not exist at this depth
      // in the adaptor stack.
      assert(static_cast<int>(CallerLayer) >= this->get_property_layer());
      return this->operator[](id.get_cluster_index(this->get_property_layer()));
    }

    template <size_t CallerOrder, size_t CallerLayer, size_t Order_ = Order>
    std::enable_if_t<(Order_ == 1) and (CallerOrder > 1),  // NOLINT
                     reference>                            // NOLINT
    operator[](const ClusterRefKey<CallerOrder, CallerLayer> & id) {
      // #BUG8486@(all) we can just use the managers function to get the
      // corresponding cluster index, no need to save this in the cluster
      return this->operator[](this->get_manager().get_atom_index(
          id.get_internal_neighbour_atom_tag()));
    }

    /**
     * Accessor for property by index for properties
     */
    reference operator[](size_t index) {
      // use tag dispatch to use the proper definition
      // of the get function
      return this->get(
          index,
          std::conditional_t<(IsStaticallySized), StaticSize, DynamicSize>{});
    }

   protected:
    void push_back(reference ref, StaticSize) {
      Value::push_in_vector(this->values, ref);
    }

    void push_back(reference ref, DynamicSize) {
      Value::push_in_vector(this->values, ref, this->get_nb_row(),
                            this->get_nb_col());
    }

    template <typename Derived>
    void push_back(const Eigen::DenseBase<Derived> & ref, StaticSize) {
      static_assert(Derived::RowsAtCompileTime == NbRow,
                    "NbRow has incorrect size.");
      static_assert(Derived::ColsAtCompileTime == NbCol,
                    "NbCol has incorrect size.");
      Value::push_in_vector(this->values, ref);
    }
    template <typename Derived>
    void push_back(const Eigen::DenseBase<Derived> & ref, DynamicSize) {
      Value::push_in_vector(this->values, ref, this->get_nb_row(),
                            this->get_nb_col());
    }

    reference get(size_t index, StaticSize) {
      return Value::get_ref(this->values[index * NbRow * NbCol]);
    }

    reference get(size_t index, DynamicSize) {
      return get_ref(this->values[index * this->get_nb_comp()]);
    }

    //! get a reference
    reference get_ref(T & value) {
      return reference(&value, this->get_nb_row(), this->get_nb_col());
    }

    std::string type_id{};
  };

}  // namespace rascal

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_HH_
