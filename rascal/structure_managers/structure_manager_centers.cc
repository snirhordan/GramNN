/**
 * @file   rascal/structure_managers/structure_manager_centers.cc
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 * @author Markus Stricker <markus.stricker@epfl.ch>
 *
 * @date   06 August 2018
 *
 * @brief Manager with atoms and centers
 *
 * Copyright  2018  Felix Musil, Markus Stricker, COSMO (EPFL), LAMMM (EPFL)
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

#include "rascal/structure_managers/structure_manager_centers.hh"

#include <fstream>
#include <iostream>
#include <numeric>

namespace rascal {

  /* ---------------------------------------------------------------------- */
  // function for setting the internal data structures
  void StructureManagerCenters::build() {
    auto && center_atoms_mask = this->get_center_atoms_mask();
    this->n_centers = center_atoms_mask.count();
    size_t ntot{
        static_cast<size_t>(this->get_positions().size() / traits::Dim)};
    this->n_ghosts = ntot - this->n_centers;

    // initialize necessary data structure
    this->atoms_index[0].clear();
    this->offsets.clear();
    internal::for_each(this->cluster_indices_container,
                       internal::ResizePropertyToZero());

    // set the references to the center atoms positions and types
    for (size_t id{0}; id < ntot; ++id) {
      if (center_atoms_mask(id)) {
        this->atoms_index[0].push_back(id);
        this->offsets.push_back(id);
      }
    }

    for (size_t id{0}; id < ntot; ++id) {
      if (not center_atoms_mask(id)) {
        this->atoms_index[0].push_back(id);
        this->offsets.push_back(id);
      }
    }

    Cell_t lat = this->atoms_object.cell;
    this->lattice.set_cell(lat);

    // Check if all atoms are inside the unit cell assuming the cell starts
    // at (0,0,0)
    auto positions_scaled = this->atoms_object.get_scaled_positions();
    double tol{1e-10};
    if ((positions_scaled.array().rowwise().minCoeff() < -tol).any() or
        (positions_scaled.array().rowwise().maxCoeff() > 1. + tol).any()) {
      std::string error{R"(Some of the atoms in the structure are not
                           inside the unit cell. Please wrap them inside it
                           with at least tolerance of 1e-10.)"};
      throw std::runtime_error(error);
    }
    // center_atoms_mask.all() is false is no centers are masked
    this->are_any_centers_masked = not center_atoms_mask.all();

    auto & atom_cluster_indices{std::get<0>(this->cluster_indices_container)};
    atom_cluster_indices.fill_sequence();

    if ((this->atoms_object.atom_types.array() >= MaxChemElements).any() or
        (this->atoms_object.atom_types.array() < 0).any()) {
      std::stringstream err_str{};
      err_str << "atom types (or atomic numbers) should be in the range [0, "
              << MaxChemElements << "].";
      throw std::runtime_error(err_str.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  // returns the number of cluster at Order=1, which is the number of atoms
  size_t StructureManagerCenters::get_nb_clusters(size_t order) const {
    if (order == 1) {
      /**
       * Note: The case for order=1 is abmiguous: one possible answer is the
       * number of centers the other possibility is the number of centers +
       * ghost atoms. Please use the get_size or get_size_with_ghosts member
       * functions
       */
      return this->n_centers + this->n_ghosts;
    } else {
      throw std::string("ERROR : Order != 1");
    }
  }
  Eigen::Matrix<int, Eigen::Dynamic, 5>
  StructureManagerCenters::get_gradients_info() const {
    throw std::logic_error(
        "Error: StructureManagerCenters does not have grandients info");
  }
  /* ---------------------------------------------------------------------- */

}  // namespace rascal
