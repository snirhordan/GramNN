/**
 * @file   rascal/structure_managers/structure_manager_collection.hh
 *
 * @author Felix Musil <felix.musil@epfl.ch>
 *
 * @date   13 Jun 2019
 *
 * @brief Implementation of a container for structure managers
 *
 * Copyright 2019 Felix Musil COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_COLLECTION_HH_
#define SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_COLLECTION_HH_

#include "rascal/math/utils.hh"
#include "rascal/structure_managers/atomic_structure.hh"
#include "rascal/structure_managers/make_structure_manager.hh"
#include "rascal/structure_managers/property.hh"
#include "rascal/structure_managers/structure_manager.hh"
#include "rascal/structure_managers/updateable_base.hh"
#include "rascal/utils/json_io.hh"
#include "rascal/utils/utils.hh"

namespace rascal {

  /**
   * A container to hold the multiple managers associated with one stack of
   * managers and allows iterations over them. Each manager stack needs to be
   * initialized stack by stack. This class provides functions to do this job.
   *
   * ManagerCollection<StructureManagerCenter, AdaptorNeighbourList,
   *                   AdaptorStrict>
   *
   * ->
   *  initializes                                     StructureManagerCenter
   *  initializes                AdaptorNeighbourList<StructureManagerCenter>
   *  initializes AdaptorStrict<<AdaptorNeighbourList<StructureManagerCenter>>
   *  and are contained in the `managers` member variable.
   *
   * @tparam Manager the root manager implementation
   * @tparam AdaptorImplementationPack the adaptors stacked on top in the order
   * of stacking
   */
  template <typename Manager,
            template <class> class... AdaptorImplementationPack>
  class ManagerCollection {
   public:
    using Self_t = ManagerCollection<Manager, AdaptorImplementationPack...>;
    using TypeHolder_t =
        StructureManagerTypeHolder<Manager, AdaptorImplementationPack...>;
    using Manager_t = typename TypeHolder_t::type;
    using ManagerPtr_t = std::shared_ptr<Manager_t>;
    using ManagerList_t = typename TypeHolder_t::type_list;
    using Hypers_t = typename Manager_t::Hypers_t;
    using traits = typename Manager_t::traits;
    using Data_t = std::vector<ManagerPtr_t>;
    using value_type = typename Data_t::value_type;
    using Matrix_t = math::Matrix_t;

   protected:
    Data_t managers{};
    Hypers_t adaptor_parameters{};

   public:
    ManagerCollection() = default;

    explicit ManagerCollection(const Hypers_t & adaptor_parameters) {
      this->adaptor_parameters = adaptor_parameters;
    }

    //! Copy constructor
    ManagerCollection(const ManagerCollection & other) = delete;

    //! Move constructor
    ManagerCollection(ManagerCollection && other) = default;

    //! Destructor
    ~ManagerCollection() = default;

    //! Copy assignment operator
    ManagerCollection & operator=(const ManagerCollection & other) = delete;

    //! Move assignment operator
    ManagerCollection & operator=(ManagerCollection && other) = default;

    /**
     * Build a new collection containing a subset of the structure managers
     * selected by selected_ids.
     *
     * @return a new manager collection
     */
    template <typename Int>
    Self_t get_subset(const std::vector<Int> & selected_ids) {
      Int idx_max{*std::max_element(selected_ids.begin(), selected_ids.end())};
      if (idx_max >= static_cast<Int>(this->size())) {
        std::stringstream error{};
        error << "selected_ids has at least one out of bound index: '"
              << idx_max
              << "' for the number of managers in the collection is '"
              << this->size() << "'.";
        throw std::runtime_error(error.str());
      }
      Self_t new_collection{this->adaptor_parameters};
      for (const auto & idx : selected_ids) {
        new_collection.add_structure(this->managers[idx]);
      }
      return new_collection;
    }

    /**
     * Give the ManagerCollection the iterator functionality using Data_t
     * functionality
     */
    using iterator = typename Data_t::iterator;
    using const_iterator = typename Data_t::const_iterator;

    iterator begin() noexcept { return this->managers.begin(); }
    const_iterator begin() const noexcept { return this->managers.begin(); }

    iterator end() noexcept { return this->managers.end(); }
    const_iterator end() const noexcept { return this->managers.end(); }

    //! set the global inputs for the adaptors
    void set_adaptor_parameters(const Hypers_t & adaptor_parameters) {
      this->adaptor_parameters = adaptor_parameters;
    }

    const Hypers_t & get_adaptors_parameters() const {
      return this->adaptor_parameters;
    }

    /**
     * Get informations necessary to the computation of gradients. It has
     * as many rows as the number gradients and they correspond to the index
     * of the structure, the central atom, the neighbor atom and their atomic
     * species.
     * The shape is (n_structures * n_centers * n_neighbor, 5) while the
     * n_neighbour is nonconstant over centers
     */
    Eigen::Matrix<int, Eigen::Dynamic, 5> get_gradients_info() const {
      if (this->managers.size() == 0) {
        throw std::runtime_error(
            R"(There are no structure to get features from)");
      }
      size_t n_neighbors{0};
      for (auto & manager : managers) {
        n_neighbors += manager->get_nb_clusters(2);
      }

      Eigen::Matrix<int, Eigen::Dynamic, 5> gradients_info(n_neighbors, 5);
      gradients_info.setZero();
      int i_row{0}, i_center{0}, i_frame{0};
      for (auto & manager : this->managers) {
        for (auto center : manager) {
          for (auto pair : center.pairs_with_self_pair()) {
            gradients_info(i_row, 0) = i_frame;
            gradients_info(i_row, 1) = i_center + center.get_atom_tag();
            gradients_info(i_row, 2) =
                i_center + pair.get_atom_j().get_atom_tag();
            gradients_info(i_row, 3) = center.get_atom_type();
            gradients_info(i_row, 4) = pair.get_atom_type();
            i_row++;
          }
        }
        i_center += static_cast<int>(manager->size());
        i_frame++;
      }
      return gradients_info;
    }

    /**
     * functions to add one or several structures to the collection
     */
    void add_structure(const Hypers_t & structure,
                       const Hypers_t & adaptor_parameters) {
      auto manager =
          make_structure_manager_stack<Manager, AdaptorImplementationPack...>(
              structure, adaptor_parameters);
      this->add_structure(manager);
    }

    void add_structure(const Hypers_t & structure) {
      auto manager =
          make_structure_manager_stack<Manager, AdaptorImplementationPack...>(
              structure, this->adaptor_parameters);
      this->add_structure(manager);
    }

    void add_structure(std::shared_ptr<Manager_t> & manager) {
      this->managers.emplace_back(manager);
    }

    /**
     * A helper function used in the python bindings. It adds empty structures
     * to then updates them with the actual atomic structure (a small workaround
     * because AtomicStructure<3> can't be put in a json object).
     *
     * @param atomic_structures the structures which are added to the managers.
     */
    void
    add_structures(const std::vector<AtomicStructure<3>> & atomic_structures) {
      Hypers_t empty_structure = Hypers_t::object();
      for (const auto & atomic_structure : atomic_structures) {
        this->add_structure(empty_structure);
        this->managers.back()->update(atomic_structure);
      }
    }

    void add_structures(const Hypers_t & structures,
                        const Hypers_t & adaptors_inputs) {
      if (not structures.is_array()) {
        throw std::runtime_error(R"(Provide the structures as an array
        (or list) of json dictionary defining the structure)");
      }
      if (structures.size() != adaptors_inputs.size()) {
        throw std::runtime_error(R"(There should be as many structures as
        adaptors_inputs)");
      }

      for (int i_structure{0}; i_structure < structures.size(); ++i_structure) {
        this->add_structure(structures[i_structure],
                            adaptors_inputs[i_structure]);
      }
    }

    void add_structures(const Hypers_t & structures) {
      if (not structures.is_array()) {
        throw std::runtime_error(R"(Provide the structures as an array
        (or list) of json dictionary defining the structure)");
      }

      for (auto & structure : structures) {
        this->add_structure(structure);
      }
    }

    /**
     * load structures from a json file
     *
     * @param filename path to the file containing the structures in
     * ase json format
     * @param start index of the first structure to include
     * @param length number of structure to include, -1 correspondons to all
     * after start
     *
     * Compatible file format are text and ubjson (binary).
     *
     * Note that start refers to 0 based indexing so 0 corresponds to the
     * first and 3 would corresponds to the 4th structure irrespective of the
     * actual indices in the file.
     */
    void add_structures(const std::string & filename, int start = 0,
                        int length = -1) {
      // important not to do brace initialization because it adds an extra
      // nesting layer
      json structures = json_io::load(filename);

      if (not structures.is_object()) {
        throw std::runtime_error(
            R"(The first level of the ase format is a dictionary with indices
                as keys to the structures)");
      }

      if (structures.count("ids") == 1) {
        // structures is in the ase format
        auto ids{structures["ids"].get<std::vector<int>>()};
        std::sort(ids.begin(), ids.end());
        ids.erase(ids.begin(), ids.begin() + start);
        if (length == -1) {
          length = ids.size();
        }
        ids.erase(ids.begin() + length, ids.end());

        for (auto & idx : ids) {
          this->add_structure(structures[std::to_string(idx)].get<Hypers_t>());
        }
      } else {
        throw std::runtime_error("The json structure format is not recognized");
      }
    }

    //! number of structure manager in the collection
    size_t size() const { return this->managers.size(); }

    /**
     * Access individual managers from the list of managers
     */
    template <typename T>
    ManagerPtr_t operator[](T index) {
      return this->managers[index]->get_shared_ptr();
    }

    template <typename T>
    ManagerPtr_t operator[](T index) const {
      return this->managers[index]->get_shared_ptr();
    }

    /**
     * get the dense feature matrix associated with the property built
     * with calculator in the elements of the manager collection.
     *
     * In the case of BlockSparseProperty the keys considered are the ones
     * present in the manager collection. In other words it means that the
     * size of the output array (nb of columns) depends on the species present
     * in the structures held by manager_collection.
     */
    template <class Calculator>
    Matrix_t get_features(const Calculator & calculator) {
      using Prop_t = typename Calculator::template Property_t<Manager_t>;

      auto property_name{this->get_calculator_name(calculator, false)};

      auto && property_ =
          *managers[0]->template get_property<Prop_t>(property_name);
      // assume inner_size is consistent for all managers
      int inner_size{property_.get_nb_comp()};

      Matrix_t features{};

      auto n_rows{this->template get_number_of_elements<Prop_t>(property_name)};

      FeatureMatrixHelper<Prop_t>::apply(this->managers, property_name,
                                         features, n_rows, inner_size);
      return features;
    }

    /**
     * @param calculator a calculator
     * @param is_gradients wether to return the name associated with the
     * features or their gradients
     * @return name of the property associated with the calculator
     */
    template <class Calculator>
    std::string get_calculator_name(const Calculator & calculator,
                                    bool is_gradients) {
      std::string property_name{};
      if (not is_gradients) {
        property_name = calculator.get_name();
      } else {
        property_name = calculator.get_gradient_name();
      }
      return property_name;
    }

    /**
     * Should only be used if calculator has BlockSparseProperty.
     * @param is_gradients wether to return the keys associated with the
     * features or their gradients
     * @return set of keys of all the BlockSparseProperty in the managers
     */
    template <class Calculator>
    auto get_keys(const Calculator & calculator, bool is_gradients = false) {
      using Prop_t = typename Calculator::template Property_t<Manager_t>;
      using Keys_t = typename Prop_t::Keys_t;

      Keys_t all_keys{};

      auto property_name{this->get_calculator_name(calculator, is_gradients)};

      for (auto & manager : this->managers) {
        auto && property =
            *manager->template get_property<Prop_t>(property_name);
        auto keys = property.get_keys();
        all_keys.insert(keys.begin(), keys.end());
      }

      return all_keys;
    }

    /**
     * @param is_gradients wether to return the number of elements associated
     * with the features or their gradients
     * @return the number of rows of the feature matrix, i.e. the number of
     * samples
     */
    template <class Prop_t>
    size_t get_number_of_elements(const std::string & property_name) {
      size_t n_elements{0};
      for (auto & manager : this->managers) {
        auto && property =
            *manager->template get_property<Prop_t>(property_name);
        n_elements += property.get_nb_item();
      }
      return n_elements;
    }

   protected:
    /**
     * Helper classes to deal with the differentiation between Property and
     * BlockSparseProperty when filling the feature matrix.
     * Fills a feature matrix with the properties contained in managers.
     */
    template <typename T>
    struct FeatureMatrixHelper {};

    template <typename T, size_t Order, int NbRow, int NbCol>
    struct FeatureMatrixHelper<Property<T, Order, Manager_t, NbRow, NbCol>> {
      using Prop_t = Property<T, Order, Manager_t, NbRow, NbCol>;
      template <class StructureManagers, class Matrix>
      static void apply(StructureManagers & managers,
                        const std::string & property_name, Matrix & features,
                        int n_rows, int inner_size) {
        features.resize(n_rows, inner_size);
        features.setZero();
        int i_row{0};
        for (auto & manager : managers) {
          auto && property =
              *manager->template get_property<Prop_t>(property_name);
          auto n_rows_manager = property.get_nb_item();
          property.fill_dense_feature_matrix(
              features.block(i_row, 0, n_rows_manager, inner_size));
          i_row += n_rows_manager;
        }
      }
    };

    template <typename T, size_t Order, typename Key>
    struct FeatureMatrixHelper<BlockSparseProperty<T, Order, Manager_t, Key>> {
      using Prop_t = BlockSparseProperty<T, Order, Manager_t, Key>;
      using Keys_t = typename Prop_t::Keys_t;

      /**
       * Fill features matrix with the representation labeled with
       * property_name present in managers using the keys in present in the
       * accross the managers.
       */
      template <class StructureManagers, class Matrix>
      static void apply(StructureManagers & managers,
                        const std::string & property_name, Matrix & features,
                        int n_rows, int inner_size) {
        Keys_t all_keys{};
        for (auto & manager : managers) {
          auto && property =
              *manager->template get_property<Prop_t>(property_name);
          auto keys = property.get_keys();
          all_keys.insert(keys.begin(), keys.end());
        }

        size_t n_cols{all_keys.size() * inner_size};
        features.resize(n_rows, n_cols);
        features.setZero();
        int i_row{0};
        for (auto & manager : managers) {
          auto && property =
              *manager->template get_property<Prop_t>(property_name);
          auto n_rows_manager = property.size();
          property.fill_dense_feature_matrix(
              features.block(i_row, 0, n_rows_manager, n_cols), all_keys);
          i_row += n_rows_manager;
        }
      }
    };
  };

}  // namespace rascal

#endif  // SRC_RASCAL_STRUCTURE_MANAGERS_STRUCTURE_MANAGER_COLLECTION_HH_
