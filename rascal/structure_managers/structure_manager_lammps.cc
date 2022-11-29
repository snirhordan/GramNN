/**
 * @file   rascal/structure_managers/structure_manager_lammps.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Apr 2018
 *
 * @brief Implementation of the neighbourhood manager for lammps
 *        neighbourhood lists
 *
 * Copyright  2018 Till Junge, COSMO (EPFL), LAMMM (EPFL)
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

#include "rascal/structure_managers/structure_manager_lammps.hh"

#include <numeric>

namespace rascal {

  /* ---------------------------------------------------------------------- */
  void StructureManagerLammps::update_self(int inum, int tot_num, int * ilist,
                                           int * numneigh, int ** firstneigh,
                                           double ** x, double ** f, int * type,
                                           double * eatom, double ** vatom) {
    // setting the class variables
    this->inum = inum;
    this->tot_num = tot_num;
    this->ilist = ilist;
    this->numneigh = numneigh;
    this->firstneigh = firstneigh;
    this->x = x;
    this->f = f;
    this->type = type;
    this->eatom = eatom;
    this->vatom = vatom;
    this->offsets.reserve(inum);
    this->offsets.resize(1);
    this->atom_index_from_atom_tag_list.clear();
    // #BUG8486@(all) it should be this->inum
    for (int i{0}; i < this->inum; ++i) {
      this->offsets.emplace_back(this->offsets[i] + this->numneigh[i]);
    }
    this->nb_pairs = std::accumulate(numneigh, numneigh + this->inum, 0);

    auto & atom_cluster_indices{std::get<0>(this->cluster_indices_container)};
    auto & pair_cluster_indices{std::get<1>(this->cluster_indices_container)};

    // #BUG8486@(all) is the solution used in make_atom_index_from_atom_tag_list
    // good enough, does ilist have huge gaps?
    // e.g. ilist = [1,5000], then the atom_index_from_atom_tag_list list will
    // have 5000 elements.
    // Also the dummy 0 values which will could undefined behaviour instead
    // necessary an error
    this->make_atom_index_from_atom_tag_list();

    atom_cluster_indices.fill_sequence();
    pair_cluster_indices.fill_sequence();
  }

  /* ---------------------------------------------------------------------- */
  /*
   * Return the number of clusters of size cluster_size.  Can only handle
   * cluster_size 1 (atoms) and cluster_size 2 (pairs).
   */
  size_t StructureManagerLammps::get_nb_clusters(int order) const {
    switch (order) {
    case 1:
      return inum;
      /**
       * Note: The case for order=1 is abmiguous: one possible answer is the
       * number of centers the other possibility is the number of centers +
       * ghost atoms. Please use the get_size or get_size_with_ghosts member
       * functions
       */
    case 2:
      return nb_pairs;
    default:
      throw std::runtime_error("Can only handle single atoms and pairs");
    }
  }
}  // namespace rascal
