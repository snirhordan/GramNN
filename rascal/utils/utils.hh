/**
 * @file   rascal/utils/utils.hh
 *
 * @author Markus Stricker <markus.stricker@epfl.ch>
 * @author Felix Musil <felix.musil@epfl.ch>
 * @date   16 Jul 2018
 *
 * @brief  utilities for rascal
 *
 * Copyright  2018 Markus Stricker, Felix Musil, COSMO (EPFL), LAMMM (EPFL)
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

#ifndef SRC_RASCAL_UTILS_UTILS_HH_
#define SRC_RASCAL_UTILS_UTILS_HH_

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace rascal {
  namespace internal {

    //! find if element appears in vector.
    template <class T, class Vec>
    bool is_element_in(const T & element, const Vec & vector) {
      Vec s_element{{element}};
      return not(std::search(vector.begin(), vector.end(), s_element.begin(),
                             s_element.end()) == vector.end());
    }

    namespace details {
      /**
       * Recursivelly go through a list of Orders and check if at least one
       * matches TargetOrder at compile time.
       */
      template <size_t TargetOrder, size_t Order, size_t... Orders>
      struct IsOrderAvailableHelper {
        static constexpr bool IsOrderAvailable = (TargetOrder == Order);

        template <bool C = IsOrderAvailable, std::enable_if_t<C, int> = 0>
        static constexpr bool get_is_order_available() {
          return true;
        }

        template <bool C = IsOrderAvailable, std::enable_if_t<not(C), int> = 0>
        static constexpr bool get_is_order_available() {
          return IsOrderAvailableHelper<TargetOrder,
                                        Orders...>::get_is_order_available();
        }
      };
      //! end of recursion
      template <size_t TargetOrder, size_t Order>
      struct IsOrderAvailableHelper<TargetOrder, Order> {
        static constexpr bool IsOrderAvailable = (TargetOrder == Order);

        template <bool C = IsOrderAvailable, std::enable_if_t<C, int> = 0>
        static constexpr bool get_is_order_available() {
          return true;
        }

        template <bool C = IsOrderAvailable, std::enable_if_t<not(C), int> = 0>
        static constexpr bool get_is_order_available() {
          return false;
        }
      };
    }  // namespace details

    /**
     * Find if TargetOrder is in the list of indices Orders.
     */
    template <size_t TargetOrder, size_t... Orders>
    static constexpr bool
    is_order_available(std::index_sequence<Orders...> /* sep*/) {
      return details::IsOrderAvailableHelper<
          TargetOrder, Orders...>::get_is_order_available();
    }

    /**
     * extract the last element of an index_sequence
     */
    template <size_t... Orders>
    constexpr size_t
    get_last_element_in_sequence(std::index_sequence<Orders...> /* sep*/) {
      constexpr std::array<size_t, sizeof...(Orders)> arr = {{Orders...}};
      return arr.back();
    }
    /**
     * Utility to check if a template parameter is iterable
     */
    template <typename... Ts>
    struct make_void {
      typedef void type;
    };
    template <typename... Ts>
    using void_t = typename make_void<Ts...>::type;

    /**
     * Checks if the type T has a begin, end, iterator and const_iterator
     * functionality.
     */
    template <class, class = void_t<>>
    struct is_iterable : std::false_type {};

    template <class T>
    struct is_iterable<T,
                       void_t<decltype(std::declval<T>().begin()),
                              decltype(std::declval<T>().end()),
                              typename T::iterator, typename T::const_iterator>>
        : std::true_type {};

    template <class, class = void_t<>>
    struct is_map : std::false_type {};

    template <class T>
    struct is_map<
        T, void_t<decltype(std::declval<T>().begin()),
                  decltype(std::declval<T>().end()), typename T::iterator,
                  typename T::const_iterator, typename T::key_type>>
        : std::true_type {};

    /**
     * Here the proper iteraror means that it is a std::Container and not
     * a std::AssociativeContainer
     */
    template <class T>
    struct is_proper_iterator {
      static constexpr bool value = !is_map<T>::value && is_iterable<T>::value;
    };

    /* ---------------------------------------------------------------------- */
    /**
     * Utility to deduce the type of a Manager with a list of Adaptors
     */
    template <typename ManagerImplementation,
              template <class> class AdaptorImplementation,
              template <class> class... AdaptorImplementationPack>
    struct AdaptorTypeStacker {
      using Manager_t = AdaptorImplementation<ManagerImplementation>;
      using type =
          typename AdaptorTypeStacker<Manager_t,
                                      AdaptorImplementationPack...>::type;
    };

    template <typename ManagerImplementation,
              template <class> class AdaptorImplementation>
    struct AdaptorTypeStacker<ManagerImplementation, AdaptorImplementation> {
      using type = AdaptorImplementation<ManagerImplementation>;
    };
    /* ---------------------------------------------------------------------- */
    /**
     * Utilities to combine Enum to flatten the nested switch cases
     * The caveat is that the enum list need to be finished with the
     * End_
     * taken from:
     * https://www.fluentcpp.com/2017/06/27/how-to-collapse-nested-switch-statements/
     * // NOLINT
     */

    //! get the underlying value of the enum
    template <typename Enum>
    constexpr size_t enum_value(Enum e) {
      return static_cast<size_t>(e);
    }

    //! compute the length of the enum assuming the last element is End_
    template <typename Enum>
    constexpr size_t enum_size() {
      return enum_value(Enum::End_);
    }

    /**
     * Combines two or three enum into a new integer making sure types are
     * properly provided by mapping the enums values to a unique integer as long
     * as the order of the enum types stays the same. Requires that the enum
     * classes have End_ enum type at the end. For example for the enums
     *
     *   Enum1 = { 0, 1, End_}
     *   Enum2 = { 0, 1, 2, End_}
     *   Enum3 = { 0, 1, End_}
     *
     * enum value tuple  ->  combined key
     *    (0,0,0)               0
     *    (1,0,0)               1
     *    (0,1,0)               2
     *    (1,1,0)               3
     *    (0,2,0)               4
     *    (1,2,0)               5
     *    (0,0,1)               6
     *    (1,0,1)               7
     *    (0,1,1)               8
     *    (1,1,1)               9
     *    (0,2,1)              10
     *    (1,2,1)              11
     *
     * Note why not in struct:
     *   template <typename Enum1, typename Enum2>
     *   struct CombineEnums {
     *     constexpr size_t operator()(Enum1 e1, Enum2 e2) {
     *       return enum_value(e1) * enum_size<Enum2>() + enum_value(e2);
     *     }
     *   };
     *
     * the above code does not compile with gcc 5 and 6 (4 and 7 works though)
     * and clang has no problem. it has to do with the implementation of the
     * standard see for more details
     * https://stackoverflow.com/questions/16493652/constexpr-not-working-if-the-function-is-declared-inside-class-scope
     * // NOLINT
     */
    template <typename Enum1, typename Enum2>
    constexpr size_t combine_enums(Enum1 e1, Enum2 e2) {
      return enum_value(e1) + enum_size<Enum1>() * enum_value(e2);
    }

    template <typename Enum1, typename Enum2, typename Enum3>
    constexpr size_t combine_enums(Enum1 e1, Enum2 e2, Enum3 e3) {
      return enum_value(e1) +
             enum_size<Enum1>() *
                 (enum_value(e2) + enum_size<Enum2>() * enum_value(e3));
    }

    /* ---------------------------------------------------------------------- */
    /**
     * Implementation of the generation of an index sequence from Min to Max
     */
    template <size_t N, size_t... Seq>
    constexpr std::index_sequence<N + Seq...>
    add_to_sequence(std::index_sequence<Seq...>) {
      return {};
    }
    //! this goes from Min to Max
    template <size_t Min, size_t Max>
    using make_index_range = decltype(
        add_to_sequence<Min>(std::make_index_sequence<Max - Min + 1>()));

    /* ---------------------------------------------------------------------- */
    //! Implementation of index_apply
    template <class F, size_t... Is>
    constexpr auto index_apply_impl(F func, std::index_sequence<Is...>) {
      return func(std::integral_constant<size_t, Is>{}...);
    }

    /**
     * index_apply and its implementation take a callable of type F and return
     * it evaluated with an index sequence starting from Min to Max
     */
    template <size_t Min, size_t Max, class F>
    constexpr auto index_apply(F func) {
      return index_apply_impl(func, make_index_range<Min, Max - 1>{});
    }

    /**
     * Extract the elements of a tuple from Min to Max
     */
    template <size_t Min, size_t Max, class Tuple>
    constexpr auto take_range(Tuple t) {
      static_assert(Min <= Max, "Min should be smaller or equal than Max");
      return index_apply<Min, Max>(
          [&](auto... Is) { return make_tuple(std::get<Is>(t)...); });
    }

    /**
     * Evaluate a callable of type F with the elements of the Tuple
     */
    template <class Tuple, class F>
    constexpr auto apply(F func, Tuple t) {
      return index_apply<0, std::tuple_size<Tuple>{}>(
          [&](auto... Is) { return func(std::get<Is>(t)...); });
    }

    /* ---------------------------------------------------------------------- */
    /**
     * Helper functions to apply a functor to all items in a tuple. The actual
     * function is ``for_each``, other functions construct the template loop
     * over the items in the tuple.
     */
    template <typename Func, typename Last>
    inline void for_each_impl(Func && f, Last && last) {
      f(last);
    }

    template <typename Func, typename Head, typename... Tail>
    inline void for_each_impl(Func && f, Head && head, Tail &&... tail) {
      f(head);
      for_each_impl(std::forward<Func>(f), tail...);
    }

    template <typename Func, size_t... Indices, typename... Args>
    inline void for_each_helper(Func && f, std::index_sequence<Indices...>,
                                std::tuple<Args...> && tup) {
      for_each_impl(std::forward<Func>(f),
                    std::forward<Args>(std::get<Indices>(tup))...);
    }

    /**
     * Utility for applying a function to individual tuple elements. `tup` is a
     * tuple that can be templated with an arbitrary number of arguments. `f` is
     * the function that should be applied to each element of the tuple.
     */
    template <typename Func, typename... Args>
    inline void for_each(std::tuple<Args...> & tup, Func && f) {
      for_each_helper(std::forward<Func>(f), std::index_sequence_for<Args...>{},
                      std::forward<std::tuple<Args...>>(tup));
    }

    /* ---------------------------------------------------------------------- */
    // useful functors to be applied to tuples somewhere else in the code

    //! Functor for resetting properties to zero size
    struct ResizePropertyToZero {
      template <typename T>
      void operator()(T & t) {
        t.clear();
      }
    };

    /**
     * Find set intersection between 2 containers.
     * @tparam Set_* std sequence container
     * @return intersections set intersection between keys_a and keys_b
     */
    template <class Set_a, class Set_b>
    auto set_intersection(const Set_a & keys_a, const Set_b & keys_b) {
      using value_type = typename Set_a::value_type;
      static_assert(std::is_same<value_type, typename Set_b::value_type>::value,
                    "Set type must be the same");
      if (keys_a.empty() or keys_b.empty()) {
        return std::set<value_type>();
      }
      std::set<value_type> set{keys_a.cbegin(), keys_a.cend()};
      std::set<value_type> intersections;
      for (const value_type & el : keys_b) {
        if (set.erase(el) > 0) {
          // if el exists in set, then 1 is returned and el is erased;
          // otherwise, 0.
          intersections.insert(el);
        }
      }
      return intersections;
    }

    /* ---------------------------------------------------------------------- */

    //! Get the demangled name of a c++ type, as returned by
    //! `typeinfo(...).name()`
    std::string type_name_demangled(const char * name);

    //! Replace all occurences of `pattern` by `replacement` in `string`
    void replace(std::string & string, const std::string & pattern,
                 const std::string & replacement);

    //! return a human readable form of the template parameter type name
    template <typename T>
    std::string type_name() {
      std::string full_name = type_name_demangled(typeid(T).name());

      replace(full_name, "rascal::", "");
      replace(full_name, "<", "_");
      replace(full_name, ">", "");
      replace(full_name, " ", "");
      replace(full_name, ",", "_");
      return full_name;
    }

    //! Reads a binary file
    //!
    //! @throw std::runtime_error if the file can't be opened
    std::vector<uint8_t> read_binary_file(const std::string & filename);

    //! Extract the extension from a filename as the charaters after the last
    //! ".". If no extension is found, return an empty string.
    std::string get_filename_extension(const std::string & filename);

  }  // namespace internal
  constexpr int ThreeD{3};
  constexpr int AtomOrder{1};
  constexpr int PairOrder{2};
  constexpr int TripletOrder{3};
  constexpr int QuadrupletOrder{4};
  constexpr int MaxChemElements{130};

}  // namespace rascal

#endif  // SRC_RASCAL_UTILS_UTILS_HH_
