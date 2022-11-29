/**
 * @file   rascal/structure_managers/property_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   03 Aug 2018
 *
 * @brief implementation of non-templated base class for Properties, Properties
 *        are atom-, pair-, triplet-, etc-related values
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_BASE_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_BASE_HH_

#include "rascal/structure_managers/structure_manager_base.hh"
#include "rascal/utils/basic_types.hh"

#include <array>
#include <string>
#include <vector>

namespace rascal {

  /**
   * Base class defintion of a ``property``, defining an interface.
   */
  class PropertyBase {
   public:
    //! Default constructor
    PropertyBase() = delete;

    //! Copy constructor
    PropertyBase(const PropertyBase & other) = delete;

    //! Move constructor
    PropertyBase(PropertyBase && other) = default;

    //! Destructor
    virtual ~PropertyBase() = default;

    //! Copy assignment operator
    PropertyBase & operator=(const PropertyBase & other) = delete;

    //! Move assignment operator
    PropertyBase & operator=(PropertyBase && other) = delete;

    //! return compile time type information
    virtual const std::string & get_type_info() const = 0;

    //! returns the number of degrees of freedom stored per cluster
    Dim_t get_nb_comp() const { return this->nb_comp; }

    //! updates the number of degrees of freedom stored per cluster
    void update_nb_comp() { this->nb_comp = this->nb_row * this->nb_col; }

    //! returns the number of rows stored per cluster
    Dim_t get_nb_row() const { return this->nb_row; }

    //! sets the number of rows stored per cluster
    void set_nb_row(const Dim_t & nb_row) {
      this->nb_row = nb_row;
      this->update_nb_comp();
    }

    //! returns the number of columns stored per cluster
    Dim_t get_nb_col() const { return this->nb_col; }

    //! sets the number of columns stored per cluster
    void set_nb_col(const Dim_t & nb_col) {
      this->nb_col = nb_col;
      this->update_nb_comp();
    }

    void set_shape(const Dim_t & nb_row, const Dim_t & nb_col) {
      this->nb_row = nb_row;
      this->nb_col = nb_col;
      this->update_nb_comp();
    }

    //! returns the cluster order
    Dim_t get_order() const { return this->order; }

    //! returns the property layer
    Dim_t get_property_layer() const { return this->property_layer; }

    //! returns the metadata string
    std::string get_metadata() const { return this->metadata; }

    /**
     * Controls the is_updated flag
     */
    bool is_updated() const { return this->updated; }

    void set_updated_status(bool is_updated) { this->updated = is_updated; }

   protected:
    //!< base-class reference to StructureManager
    StructureManagerBase & base_manager;
    Dim_t nb_col;        //!< number of columns stored
    Dim_t nb_row;        //!< number of rows stored
    Dim_t nb_comp;       //!< number of dofs stored
    const size_t order;  //!< order of the clusters
    //! layer in the stack at which property is attached
    const size_t property_layer;
    //!< e.g. a JSON formatted string
    const std::string metadata;
    //! tells if the property is in synch with the underlying structure of
    //! the structure manager
    bool updated{false};
    //! constructor
    PropertyBase(StructureManagerBase & manager, Dim_t nb_row, Dim_t nb_col,
                 size_t order, size_t layer,
                 std::string metadata = "no metadata")
        : base_manager{manager}, nb_col{nb_col}, nb_row{nb_row},
          nb_comp{nb_row * nb_col}, order{order},
          property_layer{layer}, metadata{metadata} {}
  };
}  // namespace rascal

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_PROPERTY_BASE_HH_
