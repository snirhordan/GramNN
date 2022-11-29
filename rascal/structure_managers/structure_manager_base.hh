/**
 * @file   rascal/structure_managers/structure_manager_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   06 Aug 2018
 *
 * @brief  Polymorphic base class for structure managers
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_BASE_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_BASE_HH_

#include "rascal/structure_managers/updateable_base.hh"

#include <string>

namespace rascal {

  //! polymorphic base class type for StructureManagers
  class StructureManagerBase : public Updateable {
   public:
    virtual ~StructureManagerBase() = default;
    virtual size_t nb_clusters(size_t order) const = 0;
    virtual void update_self() = 0;

   protected:
  };
}  // namespace rascal

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_BASE_HH_
