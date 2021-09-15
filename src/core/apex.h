// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
 
#pragma once

#include <fstream>
#include <iostream>
#include <stack>
#include <type_traits>
#include <csignal>
#include <map>
#include <vector>
#include <algorithm>

#include "apex_base.h"
#include "apex_fanout_tree.h"
#include "apex_nodes.h"
#include "../util/allocator.h"
#include "../tree.h"

// Whether we account for floating-point precision issues when traversing down
// APEX.
// These issues rarely occur in practice but can cause incorrect behavior.
// Turning this on will cause slight performance overhead due to extra
// computation and possibly accessing two data nodes to perform a lookup.
#define ALEX_SAFE_LOOKUP 1
#define STASH_COEFFICIENT 1.5

namespace alex {
uint64_t model_node_num = 0;
uint64_t model_node_size = 0;

template <class T, class P, class Compare = AlexCompare,
          class Alloc = my_alloc::allocator<std::pair<T, P>>,
          bool allow_duplicates = true>
class Apex : public Tree<T, P> {
  static_assert(std::is_arithmetic<T>::value, "ALEX key type must be numeric.");
  static_assert(std::is_same<Compare, AlexCompare>::value,
                "Must use AlexCompare.");

 public:
  // Value type, returned by dereferencing an iterator
  typedef std::pair<T, P> V;

  // ALEX class aliases
  typedef Apex<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef AlexModelNode<T, P, Alloc> model_node_type;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> data_node_type;

  // Forward declaration for iterators
  class Iterator;
  class ConstIterator;
  class ReverseIterator;
  class ConstReverseIterator;
  class NodeIterator;  // Iterates through all nodes with pre-order traversal

  AlexNode<T, P>* root_node_;
  model_node_type* superroot_;  // phantom node that is the root's parent
  uint32_t global_version_;
  uint32_t root_lock_;

  /* User-changeable parameters */
  struct Params {
    // When bulk loading, Alex can use provided knowledge of the expected
    // fraction of operations that will be inserts
    // For simplicity, operations are either point lookups ("reads") or inserts
    // ("writes)
    // i.e., 0 means we expect a read-only workload, 1 means write-only
    double expected_insert_frac = 1;
    // Maximum node size, in bytes. By default, 16MB.
    // Higher values result in better average throughput, but worse tail/max
    // insert latency
    int max_node_size = 1 << 24;
    // Approximate model computation: bulk load faster by using sampling to
    // train models
    bool approximate_model_computation = true;
    // Approximate cost computation: bulk load faster by using sampling to
    // compute cost
    bool approximate_cost_computation = false;
  };
  Params params_;

  /* Setting max node size automatically changes these parameters */
  struct DerivedParams {
    // The defaults here assume the default max node size of 16MB
    int max_fanout = 1 << 21;  // assumes 8-byte pointers
    int max_data_node_slots = (1 << 18) / sizeof(V);
  };
  DerivedParams derived_params_;

  /* Counters, useful for benchmarking and profiling */
  struct Stats {
    int num_keys;
    int num_model_nodes = 0;  // num model nodes
    int num_data_nodes = 0;   // num data nodes
    int num_expand_and_scales = 0;
    int num_expand_and_retrains = 0;
    int num_downward_splits = 0;
    int num_sideways_splits = 0;
    int num_model_node_expansions = 0;
    int num_model_node_splits = 0;
    long long num_downward_split_keys = 0;
    long long num_sideways_split_keys = 0;
    long long num_model_node_expansion_pointers = 0;
    long long num_model_node_split_pointers = 0;
    mutable long long num_node_lookups = 0;
    mutable long long num_lookups = 0;
    long long num_inserts = 0;
    double splitting_time = 0;
    double cost_computation_time = 0;
  };
  Stats stats_;

  /* These are for research purposes, a user should not change these */
  struct ExperimentalParams {
    // Fanout selection method used during bulk loading: 0 means use bottom-up
    // fanout tree, 1 means top-down
    int fanout_selection_method = 0;
    // Policy when a data node experiences significant cost deviation.
    // 0 means always split node in 2
    // 1 means decide between no splitting or splitting in 2
    // 2 means use a full fanout tree to decide the splitting strategy
    int splitting_policy_method = 1;
    // Splitting upwards means that a split can propagate all the way up to the
    // root, like a B+ tree
    // Splitting upwards can result in a better RMI, but has much more overhead
    // than splitting sideways
    bool allow_splitting_upwards = false;
  };
  ExperimentalParams experimental_params_;

  // Log area for SMO operation
  RootExpandLog* root_expand_log_;
  LogArray* log_; // needs to reset the log_ after this tree is created

  /* Structs used internally */

 private:
  /* Statistics related to the key domain.
   * The index can hold keys outside the domain, but lookups/inserts on those
   * keys will be inefficient.
   * If enough keys fall outside the key domain, then we expand the key domain.
   */
  struct InternalStats {
    T key_domain_min_;
    T key_domain_max_;
    int num_keys_above_key_domain;
    int num_keys_below_key_domain;
    int num_keys_at_last_right_domain_resize;
    int num_keys_at_last_left_domain_resize;
  };
  InternalStats istats_;

  /* Save the traversal path down the RMI by having a linked list of these
   * structs. */
  struct TraversalNode {
    model_node_type* node = nullptr;
    int bucketID = -1;
  };

  /* Used when finding the best way to propagate up the RMI when splitting
   * upwards.
   * Cost is in terms of additional model size created through splitting
   * upwards, measured in units of pointers.
   * One instance of this struct is created for each node on the traversal path.
   * User should take into account the cost of metadata for new model nodes
   * (base_cost). */
  struct SplitDecisionCosts {
    static constexpr double base_cost =
        static_cast<double>(sizeof(model_node_type)) / sizeof(void*);
    // Additional cost due to this node if propagation stops at this node.
    // Equal to 0 if redundant slot exists, otherwise number of new pointers due
    // to node expansion.
    double stop_cost = 0;
    // Additional cost due to this node if propagation continues past this node.
    // Equal to number of new pointers due to node splitting, plus size of
    // metadata of new model node.
    double split_cost = 0;
  };

  // At least this many keys must be outside the domain before a domain
  // expansion is triggered.
  static const int kMinOutOfDomainKeys = 100;
  // After this many keys are outside the domain, a domain expansion must be
  // triggered.
  static const int kMaxOutOfDomainKeys = 1000;
  // When the number of max out-of-domain (OOD) keys is between the min and
  // max, expand the domain if the number of OOD keys is greater than the
  // expected number of OOD due to randomness by greater than the tolereance
  // factor.
  static const int kOutOfDomainToleranceFactor = 2;

  Compare key_less_ = Compare();
  Alloc allocator_ = Alloc();

  data_node_type* debug_node = nullptr;

  /*** Constructors and setters ***/

 public:
  Apex() {
    set_default_parameters();
    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    create_superroot();
  }

  Apex(bool recover){
    if(recover){
      printf("The root node addr = %p\n", root_node_);
      this->recovery();
    }else{
      set_default_parameters();
      // Set up root as empty data node
      auto empty_data_node = new (data_node_allocator().allocate(1))
          data_node_type(key_less_, allocator_);
      empty_data_node->bulk_load(nullptr, 0);
      root_node_ = empty_data_node;
      create_superroot();
    }
  }

  ~Apex() {
    std::cout << "start the deallocate" << std::endl;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      delete_node(node_it.current());
    }
    delete_node(superroot_);
    std::cout << "end the deallocate" << std::endl;
  }

  void swap(const self_type& other) {
    std::swap(params_, other.params_);
    std::swap(derived_params_, other.derived_params_);
    std::swap(experimental_params_, other.experimental_params_);
    std::swap(istats_, other.istats_);
    std::swap(stats_, other.stats_);
    std::swap(key_less_, other.key_less_);
    std::swap(allocator_, other.allocator_);
    std::swap(superroot_, other.superroot_);
    std::swap(root_node_, other.root_node_);
  }

  void set_default_parameters(){
    root_node_ = nullptr;
    superroot_ = nullptr;
    global_version_ = 0;
    root_lock_ = 0;
    stats_.num_keys = 0;

    // Internal statistics
    istats_.key_domain_min_ = std::numeric_limits<T>::max();
    istats_.key_domain_max_ = std::numeric_limits<T>::lowest();
    istats_.num_keys_below_key_domain = 0;
    istats_.num_keys_above_key_domain = 0;
    istats_.num_keys_at_last_right_domain_resize = 0;
    istats_.num_keys_at_last_left_domain_resize = 0;

    // Allocate log area
    my_alloc::BasePMPool::AlignZAllocate(reinterpret_cast<void**>(&log_), sizeof(LogArray));
    my_alloc::BasePMPool::AlignZAllocate(reinterpret_cast<void**>(&root_expand_log_), sizeof(RootExpandLog));
  }

 private:
  // Deep copy of tree starting at given node
  AlexNode<T, P>* copy_tree_recursive(const AlexNode<T, P>* node) {
    if (!node) return nullptr;
    if (node->is_leaf_) {
      return new (data_node_allocator().allocate(1))
          data_node_type(*static_cast<const data_node_type*>(node));
    } else {
      /*
      auto node_copy = new (model_node_allocator().allocate(1))
          model_node_type(*static_cast<const model_node_type*>(node));
      */
#ifdef DRAM_MODEL
      model_node_type* node_copy;
      model_node_type::New((void**)&node_copy, static_cast<const model_node_type*>(node)->num_children_);
      new (node_copy) model_node_type(*static_cast<const model_node_type*>(node));
#else
      PMEMoid tmp;
      model_node_type::New(&tmp, static_cast<const model_node_type*>(node)->num_children_);
      auto node_copy = reinterpret_cast<model_node_type*>(pmemobj_direct(tmp));
      new (node_copy) model_node_type(*static_cast<const model_node_type*>(node));
#endif

      int cur = 0;
      while (cur < node_copy->num_children_) {
        AlexNode<T, P>* child_node = node_copy->children_[cur];
        AlexNode<T, P>* child_node_copy = copy_tree_recursive(child_node);
        int repeats = 1 << (log_2_round_down(node_copy->num_children_) - child_node_copy->local_depth_);
        for (int i = cur; i < cur + repeats; i++) {
          node_copy->children_[i] = child_node_copy;
        }
        cur += repeats;
      }
      return node_copy;
    }
  }

 public:
  // concurrency logic about holding the global lock
  inline void get_lock() {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
      while (true) {
        old_value = __atomic_load_n(&root_lock_, __ATOMIC_ACQUIRE);
        if (!(old_value & lockSet)) {
          old_value &= lockMask;
          break;
        }
      }
      new_value = old_value | lockSet;
    } while (!CAS(&root_lock_, &old_value, new_value));
  }

  inline bool try_get_lock() {
    uint32_t v = __atomic_load_n(&root_lock_, __ATOMIC_ACQUIRE);
    if (v & lockSet) {
      return false;
    }
    auto old_value = v & lockMask;
    auto new_value = v | lockSet;
    return CAS(&root_lock_, &old_value, new_value);
  }

  inline void release_lock() {
    uint32_t v = root_lock_;
    __atomic_store_n(&root_lock_, v + 1 - lockSet, __ATOMIC_RELEASE);
  }

  // When bulk loading, Alex can use provided knowledge of the expected fraction
  // of operations that will be inserts
  // For simplicity, operations are either point lookups ("reads") or inserts
  // ("writes)
  // i.e., 0 means we expect a read-only workload, 1 means write-only
  // This is only useful if you set it before bulk loading
  void set_expected_insert_frac(double expected_insert_frac) {
    assert(expected_insert_frac >= 0 && expected_insert_frac <= 1);
    params_.expected_insert_frac = expected_insert_frac;
  }

  // Maximum node size, in bytes.
  // Higher values result in better average throughput, but worse tail/max
  // insert latency.
  void set_max_node_size(int max_node_size) {
    assert(max_node_size >= sizeof(V));
    params_.max_node_size = max_node_size;
    derived_params_.max_fanout = params_.max_node_size / sizeof(void*);
    derived_params_.max_data_node_slots = params_.max_node_size / sizeof(V);
  }

  // Bulk load faster by using sampling to train models.
  // This is only useful if you set it before bulk loading.
  void set_approximate_model_computation(bool approximate_model_computation) {
    params_.approximate_model_computation = approximate_model_computation;
  }

  // Bulk load faster by using sampling to compute cost.
  // This is only useful if you set it before bulk loading.
  void set_approximate_cost_computation(bool approximate_cost_computation) {
    params_.approximate_cost_computation = approximate_cost_computation;
  }

  /*** General helpers ***/

 public:
// Return the data node that contains the key (if it exists).
// Also optionally return the traversal path to the data node.
// traversal_path should be empty when calling this function.
// The returned traversal path begins with superroot and ends with the data
// node's parent.
// BT: add concurrency now, if the concurrency race condition occurs, abort and return nullptr
#if ALEX_SAFE_LOOKUP
  forceinline data_node_type* get_leaf(
      T key) const {
    AlexNode<T, P>* cur = root_node_;
    if (cur->is_leaf_) {
      return static_cast<data_node_type*>(cur);
    }

    while (true) {
      //first test lock set  
      auto node = static_cast<model_node_type*>(cur);
      double bucketID_prediction = node->model_.predict_double(key);
      int bucketID = static_cast<int>(bucketID_prediction);
      bucketID =
          std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);
      cur = node->children_[bucketID];
      
      if (cur->is_leaf_) {
#ifndef CONCURRENCY
        stats_.num_node_lookups += cur->level_;
#endif
        auto leaf = static_cast<data_node_type*>(cur);
        // Doesn't really matter if rounding is incorrect, we just want it to be
        // fast.
        // So we don't need to use std::round or std::lround.
        int bucketID_prediction_rounded =
            static_cast<int>(bucketID_prediction + 0.5);
        double tolerance =
            10 * std::numeric_limits<double>::epsilon() * bucketID_prediction;
        // https://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison
        if (std::abs(bucketID_prediction - bucketID_prediction_rounded) <=
            tolerance) {
          if (bucketID_prediction_rounded <= bucketID_prediction) {
            //To be safe, need to snapshot the prev_leaf_
            auto prev_leaf = leaf->prev_leaf_ ;
            if (prev_leaf && prev_leaf->max_limit_ > key) {
              if(leaf->prev_leaf_ != prev_leaf) {
                return nullptr;
              }
              return prev_leaf;
            }
          } else {
            auto next_leaf = leaf->next_leaf_ ;
            if (next_leaf && next_leaf->min_limit_ <= key) {
              if(leaf->next_leaf_ != next_leaf) {
                return nullptr;
              }
              return next_leaf;
            }
          }
        }
        return leaf;
      }
    }
  }
#else
  data_node_type* get_leaf(
      T key, std::vector<TraversalNode>* traversal_path = nullptr) const {
    if (traversal_path) {
      traversal_path->push_back({superroot_, 0});
    }
    AlexNode<T, P>* cur = root_node_;

    while (!cur->is_leaf_) {
      auto node = static_cast<model_node_type*>(cur);
      int bucketID = node->model_.predict(key);
      bucketID =
          std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);
      if (traversal_path) {
        traversal_path->push_back({node, bucketID});
      }
      cur = node->children_[bucketID];
    }

    stats_.num_node_lookups += cur->level_;
    return static_cast<data_node_type*>(cur);
  }
#endif


    forceinline data_node_type* get_leaf_with_traversal(
      T key, std::vector<TraversalNode>* traversal_path) const {
    if (traversal_path) {
      traversal_path->push_back({superroot_, 0});
    }

    AlexNode<T, P>* cur = root_node_;
    if (cur->is_leaf_) {
      return static_cast<data_node_type*>(cur);
    }

    while (true) {
      //first test lock set  
      auto node = static_cast<model_node_type*>(cur);
      double bucketID_prediction = node->model_.predict_double(key);
      int bucketID = static_cast<int>(bucketID_prediction);
      bucketID =
          std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);
      
      if (traversal_path) {
        traversal_path->push_back({node, bucketID});
      }      
      cur = node->children_[bucketID];
      
      if (cur->is_leaf_) {
        auto leaf = static_cast<data_node_type*>(cur);
        // Doesn't really matter if rounding is incorrect, we just want it to be
        // fast.
        // So we don't need to use std::round or std::lround.
        int bucketID_prediction_rounded =
            static_cast<int>(bucketID_prediction + 0.5);
        double tolerance =
            10 * std::numeric_limits<double>::epsilon() * bucketID_prediction;
        // https://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison
        if (std::abs(bucketID_prediction - bucketID_prediction_rounded) <=
            tolerance) {
          if (bucketID_prediction_rounded <= bucketID_prediction) {
            //To be safe, need to snapshot the prev_leaf_
            auto prev_leaf = leaf->prev_leaf_ ;
            if (prev_leaf && prev_leaf->last_key() >= key) {
              
              if (traversal_path) {
                // Correct the traversal path
                correct_traversal_path(leaf, *traversal_path, true);
              }
              if(leaf->prev_leaf_ != prev_leaf) {
                return nullptr;
              }
              return prev_leaf;
            }
          } else {
            auto next_leaf = leaf->next_leaf_ ;
            if (next_leaf && next_leaf->first_key() <= key) {
              if (traversal_path) {
                // Correct the traversal path
                correct_traversal_path(leaf, *traversal_path, false);
              }
              if(leaf->next_leaf_ != next_leaf) {
                return nullptr;
              }
              return next_leaf;
            }
          }
        }
        return leaf;
      }
    }
  }

  // Lock parent process may fail? then retry
  // two reasons: 
  // firstly model node validation fail along the path
  // secondly try get lock may fail
  // return nullptr if the locking process fails
  // try the lock the parent of one leaf node, traversal path really does not matter in this case
forceinline bool lock_parent_node(
      T key, std::vector<TraversalNode>* traversal_path, AlexNode<T, P>* child_node, bool write_lock = false) {
    traversal_path->push_back({superroot_, 0});

    AlexNode<T, P>* cur = root_node_;
    if(cur == child_node){
      if(superroot_->get_read_lock()){ 
        // After get the lock, still need to check whether the 
        if ((child_node == static_cast<AlexNode<T,P>*>(root_node_)) && (superroot_->children_[0] == child_node))
        {
          return true;
        }
        superroot_->release_read_lock();
      }
      return false;
    }

    if(cur->is_leaf_){
      return false;
    }
    
    while (true) {
      //first test lock set  
      auto node = static_cast<model_node_type*>(cur);

      double bucketID_prediction = node->model_.predict_double(key);
      int bucketID = static_cast<int>(bucketID_prediction);
      bucketID =
          std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);

      traversal_path->push_back({node, bucketID});
      cur = node->children_[bucketID];

      // first do this detection
      if(cur == child_node){
        // Then we reverse and lock the parent
        //Reverse back to see which node to lock 
        auto cur_node = traversal_path->back().node;
        auto ID = traversal_path->back().bucketID;
        if(cur_node->children_[ID] == child_node){
          //Find target node
          if(write_lock){
            if(cur_node->get_write_lock()){// it could be locked so that it is not deleted
              return true;
            }
          }else{
            // get the read lock of the parent node
            if(cur_node->get_read_lock()){// it could be locked so that it is not deleted
              return true;
            }
          }
        }
        return false;
      }

      // In this path, we will try to correct the traversal path
      if (cur->is_leaf_) {
        // Basically, we should not enter this branch
        auto leaf = static_cast<data_node_type*>(cur);
        // Doesn't really matter if rounding is incorrect, we just want it to be
        // fast.
        // So we don't need to use std::round or std::lround.
        int bucketID_prediction_rounded =
            static_cast<int>(bucketID_prediction + 0.5);
        double tolerance =
            10 * std::numeric_limits<double>::epsilon() * bucketID_prediction;
        // https://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison
        if (std::abs(bucketID_prediction - bucketID_prediction_rounded) <=
            tolerance) {
          if (bucketID_prediction_rounded <= bucketID_prediction) {
            //To be safe, need to snapshot the prev_leaf_
            auto prev_leaf = leaf->prev_leaf_ ;
            //if (leaf->prev_leaf_ && leaf->prev_leaf_->last_key() >= key) {
            if (prev_leaf && prev_leaf->max_limit_ > key) {
                // Correct the traversal path
              model_node_type *locked_node;
              if(!lock_and_correct_traversal_path(leaf, *traversal_path, true, child_node, &locked_node, write_lock)){
                return false;
              }
              return true;
            }
          } else {
            auto next_leaf = leaf->next_leaf_ ;
            if (next_leaf && next_leaf->min_limit_ <= key) {
              model_node_type *locked_node;
              if(!lock_and_correct_traversal_path(leaf, *traversal_path, false, child_node, &locked_node, write_lock)){
                return false;
              }
              return true;
            }
          }
        }

        std::cout << "This should not occur in locking parent......." << std::endl;
        return false;
      }
    }
  }


  void get_depth_info() {
    
  }

 private:
  // Make a correction to the traversal path to instead point to the leaf node
  // that is to the left or right of the current leaf node.
  inline void correct_traversal_path(data_node_type* leaf,
                                     std::vector<TraversalNode>& traversal_path,
                                     bool left) const {
    if (left) {
      TraversalNode& tn = traversal_path.back();
      model_node_type* parent = tn.node;
      int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
      // First bucket whose pointer is to leaf
      int start_bucketID = tn.bucketID - (tn.bucketID % repeats);
      if (start_bucketID == 0) {
        // Traverse back up the traversal path to make correction
        while (start_bucketID == 0) {
          traversal_path.pop_back();
          int local_depth = parent->local_depth_;
          tn = traversal_path.back();
          parent = tn.node;
          repeats = 1 << (log_2_round_down(parent->num_children_) - local_depth);
          start_bucketID = tn.bucketID - (tn.bucketID % repeats);
        }
        int correct_bucketID = start_bucketID - 1;
        tn.bucketID = correct_bucketID;
        AlexNode<T, P>* cur = parent->children_[correct_bucketID];
        while (!cur->is_leaf_) {
          auto node = static_cast<model_node_type*>(cur);
          traversal_path.push_back({node, node->num_children_ - 1});
          cur = node->children_[node->num_children_ - 1];
        }
        assert(cur == leaf->prev_leaf_);
      } else {
        // No concurrency concern
        tn.bucketID = start_bucketID - 1;
      }
    } else {
      TraversalNode& tn = traversal_path.back();
      model_node_type* parent = tn.node;
      int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
      // First bucket whose pointer is not to leaf
      int end_bucketID = tn.bucketID - (tn.bucketID % repeats) + repeats;
      if (end_bucketID == parent->num_children_) {
        // Traverse back up the traversal path to make correction
        while (end_bucketID == parent->num_children_) {
          traversal_path.pop_back();
          int local_depth = parent->local_depth_;
          tn = traversal_path.back();
          parent = tn.node;
          repeats = 1 << (log_2_round_down(parent->num_children_) - local_depth);
          end_bucketID = tn.bucketID - (tn.bucketID % repeats) + repeats;
        }
        int correct_bucketID = end_bucketID;
        tn.bucketID = correct_bucketID;
        AlexNode<T, P>* cur = parent->children_[correct_bucketID];
        while (!cur->is_leaf_) {
          auto node = static_cast<model_node_type*>(cur);
          traversal_path.push_back({node, 0});
          cur = node->children_[0];
        }
        assert(cur == leaf->next_leaf_);
      } else {
        tn.bucketID = end_bucketID;
      }
    }
  }

  // Correct the traversal path and also lock the parent
  inline bool lock_and_correct_traversal_path(data_node_type* leaf,
                                     std::vector<TraversalNode>& traversal_path,
                                     bool left, AlexNode<T, P>* child_node, model_node_type **locked_node, bool write_lock = false) const {
    //Note that the target child node should be changed
    if (left) {
      TraversalNode& tn = traversal_path.back();
      model_node_type* parent = tn.node;
      int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
      // First bucket whose pointer is to leaf
      int start_bucketID = tn.bucketID - (tn.bucketID % repeats);
      if (start_bucketID == 0) {
        // Traverse back up the traversal path to make correction
        while (start_bucketID == 0) {
          traversal_path.pop_back();
          int local_depth = parent->local_depth_;
          tn = traversal_path.back();
          parent = tn.node;
          repeats = 1 << (log_2_round_down(parent->num_children_) - local_depth);
          start_bucketID = tn.bucketID - (tn.bucketID % repeats);
        }
        int correct_bucketID = start_bucketID - 1;
        tn.bucketID = correct_bucketID;
        AlexNode<T, P>* cur = parent->children_[correct_bucketID];
        while (!cur->is_leaf_) {
          auto node = static_cast<model_node_type*>(cur);
          traversal_path.push_back({node, node->num_children_ - 1});
          cur = node->children_[node->num_children_ - 1];
        }
        assert(cur == leaf->prev_leaf_);
      } else {
        tn.bucketID = start_bucketID - 1;
      }
    } else {
      TraversalNode& tn = traversal_path.back();
      model_node_type* parent = tn.node;
      int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
      // First bucket whose pointer is not to leaf
      int end_bucketID = tn.bucketID - (tn.bucketID % repeats) + repeats;
      if (end_bucketID == parent->num_children_) {
        // Traverse back up the traversal path to make correction
        while (end_bucketID == parent->num_children_) {
          traversal_path.pop_back();
          int local_depth = parent->local_depth_;
          tn = traversal_path.back();
          parent = tn.node;
          repeats = 1 << (log_2_round_down(parent->num_children_) - local_depth);
          end_bucketID = tn.bucketID - (tn.bucketID % repeats) + repeats;
        }
        int correct_bucketID = end_bucketID;
        tn.bucketID = correct_bucketID;
        AlexNode<T, P>* cur = parent->children_[correct_bucketID];
        while (!cur->is_leaf_) {
          auto node = static_cast<model_node_type*>(cur);
          traversal_path.push_back({node, 0});
          cur = node->children_[0];
        }
        assert(cur == leaf->next_leaf_);
      } else {
        tn.bucketID = end_bucketID;
      }
    }

    int count = 0;
    auto iter = traversal_path.rbegin();
    for(; iter != traversal_path.rend(); ++iter){
      auto cur_node = (*iter).node;
      auto ID = (*iter).bucketID;
      if(cur_node->children_[ID] == child_node){
        //Find target node
        if(write_lock){
          if(cur_node->get_write_lock()){// it could be locked so that it is not deleted
            if (cur_node->children_[ID] == child_node)
            {
              *locked_node = cur_node;
              if(count){
                traversal_path.erase(traversal_path.end() - count, traversal_path.end());
              }
              return true;
            }
            cur_node->release_write_lock();
          }
        }else{
          if(cur_node->get_read_lock()){// it could be locked so that it is not deleted
            if (cur_node->children_[ID] == child_node)
            {
              *locked_node = cur_node;
              if(count){
                traversal_path.erase(traversal_path.end() - count, traversal_path.end());
              }
              return true;
            }
            cur_node->release_read_lock();
          }
        }
        return false;
      }
      ++count;
    }
    return false;    
  }

  // Return left-most data node
  data_node_type* first_data_node() const {
    AlexNode<T, P>* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_[0];
    }
    return static_cast<data_node_type*>(cur);
  }

  // Return right-most data node
  data_node_type* last_data_node() const {
    AlexNode<T, P>* cur = root_node_;

    while (!cur->is_leaf_) {
      auto node = static_cast<model_node_type*>(cur);
      cur = node->children_[node->num_children_ - 1];
    }
    return static_cast<data_node_type*>(cur);
  }

  // Returns minimum key in the index
  T get_min_key() const { return first_data_node()->first_key(); }

  // Returns maximum key in the index
  T get_max_key() const { return last_data_node()->last_key(); }

  // Link all data nodes together. Used after bulk loading.
  void link_all_data_nodes() {
    data_node_type* prev_leaf = nullptr;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      AlexNode<T, P>* cur = node_it.current();
      if (cur->is_leaf_) {
        auto node = static_cast<data_node_type*>(cur);
        if (prev_leaf != nullptr) {
          prev_leaf->next_leaf_ = node;
          node->prev_leaf_ = prev_leaf;
        }
        prev_leaf = node;
      }
    }    
  }

  // Link the new data nodes together when old data node is replaced by two new
  // data nodes.
  void link_data_nodes(data_node_type* old_leaf,
                       data_node_type* left_leaf, data_node_type* right_leaf) {
    //lock prev_leaf 
    data_node_type *prev_leaf;
    do{
      prev_leaf = old_leaf->prev_leaf_;
      if(prev_leaf == nullptr) break;
      if(prev_leaf->try_get_link_lock()){
        if(prev_leaf == old_leaf->prev_leaf_){ //ensure to lock the correct node
          prev_leaf->next_leaf_ = left_leaf;
          my_alloc::BasePMPool::Persist(&(prev_leaf->next_leaf_), sizeof(prev_leaf->next_leaf_));
          left_leaf->prev_leaf_ = prev_leaf;
          my_alloc::BasePMPool::Persist(&(left_leaf->prev_leaf_), sizeof(left_leaf->prev_leaf_));
          //prev_leaf->release_link_lock();
          break;
        }else{
          prev_leaf->release_link_lock();
        }
      }
    }while (true); 
    
    //lock cur_leaf_
    right_leaf->get_link_lock();
    old_leaf->get_link_lock();
    auto next_leaf = old_leaf->next_leaf_;
    if(next_leaf != nullptr){
      right_leaf->next_leaf_ = next_leaf;
      my_alloc::BasePMPool::Persist(&(right_leaf->next_leaf_), sizeof(right_leaf->next_leaf_));
      next_leaf->prev_leaf_ = right_leaf;
      my_alloc::BasePMPool::Persist(&(next_leaf->prev_leaf_), sizeof(next_leaf->prev_leaf_));
    }
    //old_leaf->release_link_lock();
  }

   void link_data_nodes_without_lock(data_node_type* old_leaf,
                       data_node_type* left_leaf, data_node_type* right_leaf) {
    data_node_type *prev_leaf = old_leaf->prev_leaf_;
    prev_leaf->next_leaf_ = left_leaf;
    my_alloc::BasePMPool::Persist(&(prev_leaf->next_leaf_), sizeof(prev_leaf->next_leaf_));
    left_leaf->prev_leaf_ = prev_leaf;
    my_alloc::BasePMPool::Persist(&(left_leaf->prev_leaf_), sizeof(left_leaf->prev_leaf_));

    data_node_type* next_leaf = old_leaf->next_leaf_;
    right_leaf->next_leaf_ = next_leaf;
    my_alloc::BasePMPool::Persist(&(right_leaf->next_leaf_), sizeof(next_leaf));
    next_leaf->prev_leaf_ = right_leaf;
    my_alloc::BasePMPool::Persist(&(next_leaf->prev_leaf_), sizeof(right_leaf));
  }

  void release_link_locks_for_split(data_node_type* left_leaf, data_node_type* right_leaf){
    data_node_type *prev_leaf = left_leaf->prev_leaf_;
    if (prev_leaf != nullptr)
    {
      prev_leaf->release_link_lock();
    }
    right_leaf->release_link_lock();
  }

  void link_resizing_data_nodes(data_node_type* old_leaf,
                       data_node_type* new_leaf) {
    //lock prev_leaf 
    data_node_type *prev_leaf;
    do{
      prev_leaf = old_leaf->prev_leaf_;
      if(prev_leaf == nullptr) break;
      if(prev_leaf->try_get_link_lock()){
        if(prev_leaf == old_leaf->prev_leaf_){ //ensure to lock the correct node
          prev_leaf->next_leaf_ = new_leaf;
          my_alloc::BasePMPool::Persist(&(prev_leaf->next_leaf_), sizeof(prev_leaf->next_leaf_));
          new_leaf->prev_leaf_ = prev_leaf;
          my_alloc::BasePMPool::Persist(&(new_leaf->prev_leaf_), sizeof(new_leaf->prev_leaf_));
          //prev_leaf->release_link_lock();
          break;
        }else{
          prev_leaf->release_link_lock();
        }
      }
    }while (true); 
    
    //lock cur_leaf_
    new_leaf->get_link_lock();
    old_leaf->get_link_lock();
    auto next_leaf = old_leaf->next_leaf_;
    if(next_leaf != nullptr){
      new_leaf->next_leaf_ = next_leaf;
      my_alloc::BasePMPool::Persist(&(new_leaf->next_leaf_), sizeof(next_leaf));
      next_leaf->prev_leaf_ = new_leaf;
      my_alloc::BasePMPool::Persist(&(next_leaf->prev_leaf_), sizeof(new_leaf));
    }
    //old_leaf->release_link_lock();
  }

  void release_link_locks_for_resizing(data_node_type* new_leaf){
    data_node_type *prev_leaf = new_leaf->prev_leaf_;
    if (prev_leaf != nullptr)
    {
      prev_leaf->release_link_lock();
    }
    new_leaf->release_link_lock();
  }

  void link_resizing_data_nodes_without_lock(data_node_type *old_leaf, data_node_type* new_leaf){
    data_node_type *prev_leaf = old_leaf->prev_leaf_;
    prev_leaf->next_leaf_ = new_leaf;
    my_alloc::BasePMPool::Persist(&(prev_leaf->next_leaf_), sizeof(prev_leaf->next_leaf_));
    new_leaf->prev_leaf_ = prev_leaf;
    my_alloc::BasePMPool::Persist(&(new_leaf->prev_leaf_), sizeof(new_leaf->prev_leaf_));

    data_node_type* next_leaf = old_leaf->next_leaf_;
    new_leaf->next_leaf_ = next_leaf;
    my_alloc::BasePMPool::Persist(&(new_leaf->next_leaf_), sizeof(next_leaf));
    next_leaf->prev_leaf_ = new_leaf;
    my_alloc::BasePMPool::Persist(&(next_leaf->prev_leaf_), sizeof(new_leaf));
  } 

  /*** Allocators and comparators ***/

 public:
  Alloc get_allocator() const { return allocator_; }

  Compare key_comp() const { return key_less_; }

 private:
  typename model_node_type::alloc_type model_node_allocator() {
    return typename model_node_type::alloc_type(allocator_);
  }

  typename data_node_type::alloc_type data_node_allocator() {
    return typename data_node_type::alloc_type(allocator_);
  }

  typename model_node_type::pointer_alloc_type pointer_allocator() {
    return typename model_node_type::pointer_alloc_type(allocator_);
  }

  // now we allow lock-free search
  void delete_node(AlexNode<T, P>* node) {
    if (node == nullptr) {
      return;
    } else if (node->is_leaf_) {
      data_node_allocator().destroy(static_cast<data_node_type*>(node));
      data_node_allocator().deallocate(static_cast<data_node_type*>(node), 1);
    } else {
      model_node_allocator().destroy(static_cast<model_node_type*>(node));
      model_node_allocator().deallocate(static_cast<model_node_type*>(node), 1);
    }
  }

  void safe_delete_node(AlexNode<T, P>* node) {
    if (node == nullptr) {
      return;
    } else if (node->is_leaf_) { 
      data_node_type::Free(reinterpret_cast<data_node_type*>(node));
    } else {
      my_alloc::BasePMPool::SafeFree((void*)node);
    }
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const T& a, const K& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Bulk loading ***/

 public:
  // values should be the sorted array of key-payload pairs.
  // The number of elements should be num_keyfs.
  // The index must be empty when calling this method.
  void bulk_load(const V values[], int num_keys) {
    if (stats_.num_keys > 0 || num_keys <= 0) {
      return;
    }
    delete_node(root_node_);  // delete the empty root node from constructor

    stats_.num_keys = num_keys;
    log_->clear_log();

    // Build temporary root model, which outputs a CDF in the range [0, 1]
    //root_node_ =
    //    new (model_node_allocator().allocate(1)) model_node_type(0, allocator_);
    PMEMoid tmp;
    model_node_type::New(&tmp, 0);
    new (reinterpret_cast<model_node_type*>(pmemobj_direct(tmp))) model_node_type(0, allocator_);
    root_node_ = reinterpret_cast<AlexNode<T,P>*>(pmemobj_direct(tmp));

    T min_key = values[0].first;
    T max_key = values[num_keys - 1].first;
    root_node_->model_.a_ = 1.0 / (max_key - min_key);
    root_node_->model_.b_ = -1.0 * min_key * root_node_->model_.a_;

    // Compute cost of root node
    LinearModel<T> root_data_node_model;
    data_node_type::build_model(values, num_keys, &root_data_node_model,
                                params_.approximate_model_computation);
    DataNodeStats stats;
    root_node_->cost_ = data_node_type::compute_expected_cost(
        values, num_keys, data_node_type::kInitDensity_,
        params_.expected_insert_frac, &root_data_node_model,
        params_.approximate_cost_computation, &stats);

    // Recursively bulk load
    bulk_load_node(values, num_keys, root_node_, num_keys, static_cast<double>(min_key), static_cast<double>(max_key) ,
                   &root_data_node_model);

    if (root_node_->is_leaf_) {
      static_cast<data_node_type*>(root_node_)
          ->expected_avg_search_cost_ = stats.num_search_cost;
      static_cast<data_node_type*>(root_node_)->expected_avg_insert_cost_ =
          stats.num_insert_cost;
    }

    create_superroot();
    update_superroot_key_domain(&values[0].first, &values[num_keys - 1].first);
    link_all_data_nodes();

    if(root_node_->is_leaf_) {
      //FIXME (BT): crash consistency
      MyLog my_log;
      std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;
      split_sideways_downwards_without_parent(reinterpret_cast<data_node_type*>(root_node_), 1, used_fanout_tree_nodes,
      true, min_key, &my_log, true, 0.05);
      auto log = &(my_log.smo_log_);
      // Get left children
      data_node_type* left = reinterpret_cast<data_node_type*>(pmemobj_direct(log->reserved_child_[0]));
      left->set_invalid_key(left->max_limit_+1);

      // Get right children
      data_node_type* right = reinterpret_cast<data_node_type*>(pmemobj_direct(log->reserved_child_[1]));
      left->set_invalid_key(left->min_limit_-1);
    }

    my_alloc::BasePMPool::Persist(this, sizeof(*this));
  }

 private:
  // Only call this after creating a root node
  void create_superroot() {
    if (!root_node_) return;
    delete_node(superroot_);
    PMEMoid tmp;
    model_node_type::New(&tmp, 1);
    superroot_ = reinterpret_cast<model_node_type*>(pmemobj_direct(tmp));
    new (superroot_)  model_node_type(static_cast<short>(root_node_->level_ - 1), allocator_);
    update_superroot_pointer();
  }

  // Updates the key domain based on the min/max keys and retrains the model.
  // Should only be called immediately after bulk loading or when the root node
  // is a data node.
  // NO need for persistency, re-update the key domain
  void update_superroot_key_domain(const T* min_key = nullptr, const T* max_key = nullptr) {
    get_lock();
    assert(stats_.num_inserts == 0 || root_node_->is_leaf_);
    if(min_key == nullptr){
      istats_.key_domain_min_ = get_min_key();
    }else{
      istats_.key_domain_min_ = *min_key;
    }
    if(max_key == nullptr){
      istats_.key_domain_max_ = get_max_key();
    }else{
      istats_.key_domain_max_ = *max_key;
    }    
    istats_.num_keys_at_last_right_domain_resize = stats_.num_keys;
    istats_.num_keys_at_last_left_domain_resize = stats_.num_keys;
    istats_.num_keys_above_key_domain = 0;
    istats_.num_keys_below_key_domain = 0;
    my_alloc::BasePMPool::Persist(&istats_, sizeof(istats_));
    superroot_->model_.a_ =
        1.0 / (istats_.key_domain_max_ - istats_.key_domain_min_);
    superroot_->model_.b_ =
        -1.0 * istats_.key_domain_min_ * superroot_->model_.a_;
    release_lock();
  }

  void update_superroot_pointer() {
    superroot_->children_[0] = root_node_;
    my_alloc::BasePMPool::Persist(superroot_->children_, sizeof(root_node_));
    superroot_->level_ = static_cast<short>(root_node_->level_ - 1);
    my_alloc::BasePMPool::Persist(&(superroot_->level_), sizeof(superroot_->level_));
  }

  // Recursively bulk load a single node.
  // Assumes node has already been trained to output [0, 1), has cost.
  // Figures out the optimal partitioning of children.
  // node is trained as if it's a model node.
  // data_node_model is what the node's model would be if it were a data node of
  // dense keys.
  void bulk_load_node(const V values[], int num_keys, AlexNode<T, P>*& node,
                      int total_keys, double min_limit, double max_limit, 
                      const LinearModel<T>* data_node_model = nullptr, double overflow_frac = 0) {
    // Automatically convert to data node when it is impossible to be better
    // than current cost
    if (num_keys <= derived_params_.max_data_node_slots *
                        data_node_type::kInitDensity_ &&
        (node->cost_ < kNodeLookupsWeight || node->model_.a_ == 0)) {

      stats_.num_data_nodes++;
      auto data_node = new (data_node_allocator().allocate(1))
          data_node_type(node->level_, derived_params_.max_data_node_slots,
                         key_less_, allocator_);
      data_node->min_limit_ = min_limit;
      data_node->max_limit_ = max_limit;
      if((0 >= min_limit) && ( 0 < max_limit)){
        data_node->invalid_key_ = data_node->max_limit_ + 1;
      }
      int data_capacity = static_cast<int>(num_keys / data_node_type::kMaxDensity_);
      double overflow = data_node->compute_overflow_frac(values, num_keys, data_capacity, data_node_model);
      double stash_frac = std::max(0.05, std::min(0.3, overflow * STASH_COEFFICIENT));
      data_node->bulk_load(values, num_keys, data_node_model,
                           params_.approximate_model_computation, stash_frac);
      data_node->cost_ = node->cost_;
      delete_node(node);
      node = data_node;
      my_alloc::BasePMPool::Persist(data_node, sizeof(data_node_type));
      return;
    }

    // Use a fanout tree to determine the best way to divide the key space into
    // child nodes
    std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;
    std::pair<int, double> best_fanout_stats;
    if (experimental_params_.fanout_selection_method == 0) {
      int max_data_node_keys = static_cast<int>(
          derived_params_.max_data_node_slots * data_node_type::kInitDensity_);
      best_fanout_stats = fanout_tree::find_best_fanout_bottom_up<T, P>(
          values, num_keys, node, total_keys, used_fanout_tree_nodes,
          derived_params_.max_fanout, max_data_node_keys, params_.expected_insert_frac,
          params_.approximate_model_computation,
          params_.approximate_cost_computation, key_less_);
    } else if (experimental_params_.fanout_selection_method == 1) {
      best_fanout_stats = fanout_tree::find_best_fanout_top_down<T, P>(
          values, num_keys, node, total_keys, used_fanout_tree_nodes,
          derived_params_.max_fanout, params_.expected_insert_frac,
          params_.approximate_model_computation,
          params_.approximate_cost_computation, key_less_);
    }
    int best_fanout_tree_depth = best_fanout_stats.first;
    double best_fanout_tree_cost = best_fanout_stats.second;

   
    // Decide whether this node should be a model node or data node
    if (best_fanout_tree_cost < node->cost_ ||
        num_keys > derived_params_.max_data_node_slots *
                       data_node_type::kInitDensity_) {      
      if (best_fanout_tree_depth == 0) {
        // slightly hacky: we assume this means that the node is relatively
        // uniform but we need to split in
        // order to satisfy the max node size, so we compute the fanout that
        // would satisfy that condition
        // in expectation
        best_fanout_tree_depth =
            static_cast<int>(std::log2(static_cast<double>(num_keys) /
                                       derived_params_.max_data_node_slots)) +
            1;
        used_fanout_tree_nodes.clear();
        int max_data_node_keys = static_cast<int>(
            derived_params_.max_data_node_slots * data_node_type::kInitDensity_);
        fanout_tree::compute_level<T, P>(
            values, num_keys, node, total_keys, used_fanout_tree_nodes,
            best_fanout_tree_depth, max_data_node_keys, params_.expected_insert_frac,
            params_.approximate_model_computation,
            params_.approximate_cost_computation);
      }
      // create the model node and also children array
      // Convert to model node based on the output of the fanout tree
      int fanout = 1 << best_fanout_tree_depth;
      stats_.num_model_nodes++;
      PMEMoid tmp;
      model_node_type::New(&tmp, fanout);
      auto model_node = reinterpret_cast<model_node_type*>(pmemobj_direct(tmp));
      new (model_node) model_node_type(node->level_, allocator_); //
      //  auto model_node = new (model_node_allocator().allocate(1))
      //    model_node_type(node->level_, allocator_);
      model_node->model_.a_ = node->model_.a_ * fanout;
      model_node->model_.b_ = node->model_.b_ * fanout;
      model_node->num_children_ = fanout;

      // Instantiate all the child nodes and recurse
      int cur = 0;
      for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
        PMEMoid tmp;
        model_node_type::New(&tmp, 0);
        auto child_node = reinterpret_cast<model_node_type*>(pmemobj_direct(tmp));
        new (child_node) model_node_type(node->level_ + 1, allocator_);
        //auto child_node = new (model_node_allocator().allocate(1))
        //    model_node_type(static_cast<short>(node->level_ + 1), allocator_);
        child_node->cost_ = tree_node.cost;
        child_node->local_depth_ = tree_node.level;
        int repeats = 1 << (best_fanout_tree_depth - tree_node.level);
  
        double left_value = static_cast<double>(cur) / fanout;
        double right_value = static_cast<double>(cur + repeats) / fanout;
        double left_boundary = (left_value - node->model_.b_) / node->model_.a_;
        double right_boundary =
            (right_value - node->model_.b_) / node->model_.a_;
        child_node->model_.a_ = 1.0 / (right_boundary - left_boundary);
        child_node->model_.b_ = -child_node->model_.a_ * left_boundary;
        model_node->children_[cur] = child_node;
        LinearModel<T> child_data_node_model(tree_node.a, tree_node.b);

        bulk_load_node(values + tree_node.left_boundary,
                       tree_node.right_boundary - tree_node.left_boundary,
                       model_node->children_[cur], total_keys, left_boundary, right_boundary,
                       &child_data_node_model);
        
        model_node->children_[cur]->local_depth_ = tree_node.level;
        if (model_node->children_[cur]->is_leaf_) {
          static_cast<data_node_type*>(model_node->children_[cur])
              ->expected_avg_search_cost_ =
              tree_node.expected_avg_search_cost;
          static_cast<data_node_type*>(model_node->children_[cur])
              ->expected_avg_insert_cost_ = tree_node.expected_avg_insert_cost;
        }
        for (int i = cur + 1; i < cur + repeats; i++) {
          model_node->children_[i] = model_node->children_[cur];
        }
        cur += repeats;
      }

      delete_node(node);
      node = model_node;
      my_alloc::BasePMPool::Persist(model_node, sizeof(model_node_type));
    } else {
      // Convert to data node
      stats_.num_data_nodes++;
      auto data_node = new (data_node_allocator().allocate(1))
          data_node_type(node->level_, derived_params_.max_data_node_slots,
                         key_less_, allocator_);
      data_node->min_limit_ = min_limit;
      data_node->max_limit_ = max_limit;
      if((0 >= min_limit) && ( 0 < max_limit)){
        data_node->invalid_key_ = data_node->max_limit_ + 1;
      }
      int data_capacity = static_cast<int>(num_keys / data_node_type::kMaxDensity_);
      auto overflow = data_node->compute_overflow_frac(values, num_keys, data_capacity, data_node_model);
      double stash_frac = std::max(0.05, std::min(0.3, overflow * STASH_COEFFICIENT));
      data_node->bulk_load(values, num_keys, data_node_model,
                           params_.approximate_model_computation, stash_frac);
      data_node->cost_ = node->cost_;

      delete_node(node);
      node = data_node;
      my_alloc::BasePMPool::Persist(data_node, sizeof(data_node_type));
    }
  }

  // Caller needs to set the level, duplication factor, and neighbor pointers of
  // the returned data node
  data_node_type* bulk_load_leaf_node_from_existing(
      data_node_type* existing_node, int left, int right,
      bool compute_cost = true, const fanout_tree::FTNode* tree_node = nullptr,
      bool reuse_model = false, bool keep_left = false,
      bool keep_right = false, bool build_sorted_node = false, PMEMoid* reserved_ptr = nullptr, double stash_frac = 0.05, bool is_left = false) {
      // the node is unsorted, thus needs to first make it sorted
      if(build_sorted_node) existing_node->build_sorted_slots();
      data_node_type *node;
      if(reserved_ptr != nullptr){
        my_alloc::BasePMPool::Allocate(reserved_ptr, sizeof(data_node_type));
        node = reinterpret_cast<data_node_type*>(pmemobj_direct(*reserved_ptr));
        new (node) data_node_type(existing_node->invalid_key_, key_less_, allocator_);
      }else{
        node = new (data_node_allocator().allocate(1))
            data_node_type(existing_node->invalid_key_, key_less_, allocator_);
      }
      if (tree_node) {
        // Use the model and num_keys saved in the tree node so we don't have to
        // recompute it
        LinearModel<T> precomputed_model(tree_node->a, tree_node->b);
        node->bulk_load_from_existing(existing_node, left, right, keep_left,
                                      keep_right, &precomputed_model,
                                      tree_node->num_keys, build_sorted_node, stash_frac);
      } else if (reuse_model) {
        // Use the model from the existing node
        // Assumes the model is accurate
        double mid_limit = existing_node->min_limit_ + (existing_node->max_limit_ - existing_node->min_limit_) / 2.0;
        int mid_bound = existing_node->get_position_in_PA(mid_limit);
        int real_left, real_right;
        if(is_left){
          real_left = 0;
          real_right = mid_bound;
        }else{
          real_left = mid_bound;
          real_right = existing_node->data_capacity_;
        }
        int num_actual_keys = existing_node->num_keys_in_range(left, right, true);
        LinearModel<T> precomputed_model(existing_node->model_);
        precomputed_model.b_ -= real_left;
        precomputed_model.expand(static_cast<double>(num_actual_keys) /
                                (real_right - real_left));

        node->bulk_load_from_existing(existing_node, left, right, keep_left,
                                      keep_right, &precomputed_model,
                                      num_actual_keys, build_sorted_node, stash_frac);
      } else {
        node->bulk_load_from_existing(existing_node, left, right, keep_left,
                                      keep_right, nullptr, -1, build_sorted_node, stash_frac);
      }
      node->max_slots_ = derived_params_.max_data_node_slots;
      my_alloc::BasePMPool::Persist(&node->max_slots_, sizeof(node->max_slots_));
      if (compute_cost) {
        node->cost_ = node->compute_expected_cost(existing_node->frac_inserts());
        my_alloc::BasePMPool::Persist(&node->cost_, sizeof(node->cost_));
      }
      return node;
  }

  /*** Lookup ***/

 public:
  // Looks for an exact match of the key
  bool search(const T& key, P* payload, bool epoch = false) const { 
    if(epoch) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      return search_unsafe(key, payload);
    }
    return search_unsafe(key, payload);
  }

  inline bool search_unsafe(const T& key, P* payload) const {
    do{
      data_node_type* leaf = get_leaf(key);
      if(leaf->local_version_ != global_version_){
        leaf->recover_node(global_version_);
      }
      bool found = false;
      auto ret_flag = leaf->find_payload(key, payload, &found);
      if(ret_flag == true) return found; // ret_flag == true means no concurrency conlict occurs
    }while(true);
  }

  /*** Insert ***/

 public:
  void print_min_max(){
        printf("official min in the tree = %.20f\n", istats_.key_domain_min_);
        printf("official max in the tree = %.20f\n", istats_.key_domain_max_);
  }

  // Get a set of log descriptors from the global log
  // local_log_ is the log sets specific to this thread
  class LocalLog{
    public:
      LogArray *log_array_;
      MyLog *local_log_; // assigned local log
      uint64_t local_index_;

      LocalLog(LogArray *log_array){
        //Initialize from the pool array.
        log_array_ = log_array;
        local_log_ = log_array_->assign_log(&local_index_);
      }

      ~LocalLog(){
        log_array_->return_log(local_index_);
      }
  };

  bool insert(const T& key, const P& payload, bool epoch = false) {
    if(epoch) {
        auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
        return insert_unsafe(key, payload);
    }
    return insert_unsafe(key, payload);
  }

  // This will NOT do an update of an existing key.
  // Insert does not happen if duplicate is
  // found.
  bool insert_unsafe(const T& key, const P& payload) {
    // If enough keys fall outside the key domain, expand the root to expand the
    // key domain
RETRY:
    if (key > istats_.key_domain_max_) {
      istats_.num_keys_above_key_domain++;
      if (should_expand_right()) {
        expand_root(key, false);  // expand to the right
      }
    } else if (key < istats_.key_domain_min_) {
      istats_.num_keys_below_key_domain++;
      if (should_expand_left()) {
        expand_root(key, true);  // expand to the left
      }
    }
  
    thread_local LocalLog my_log(log_); // The log descriptor
    data_node_type* leaf = get_leaf(key);
    // First test it whether it needs to be recoverred
    if(leaf->local_version_ != global_version_){
      leaf->recover_node(global_version_);
    }

    std::pair<int, double> ret = leaf->insert(key, payload, &(my_log.local_log_->overflow_desc_));
    int fail = ret.first;
    double stash_frac = ret.second;

    // If no insert, figure out what to do with the data node to decrease the
    // cost
    if (fail) {
      if(fail == -1){ // Duplicate key detected
        return false;
      }

      if (fail == 4)
      {
        goto RETRY; // The operation is in a SMO, need retry
      }

      if (fail == 5) // Data node resizing
      {
        // Directly resize the current node
        // Resize this node and install to the parent
        // 0. Start logging
        ResizeLog *resize_log = &(my_log.local_log_->resize_log_);
        resize_log->progress_ = 1;
        T insert_key = key;
        memcpy(reinterpret_cast<void*>(resize_log->key_), reinterpret_cast<void*>(&insert_key), sizeof(key));
        resize_log->cur_node_ = pmemobj_oid(reinterpret_cast<void*>(leaf)); // No need to flush; when allocate new node, cur_node_ will also be flushed into the PM
        
        // 1. Allocate new node
        data_node_type::New_from_existing(&(resize_log->new_node_), leaf);
        auto node = reinterpret_cast<data_node_type*>(pmemobj_direct(resize_log->new_node_));

        // 2. Rehash from old node to new node
        bool keep_left = leaf->is_append_mostly_right();
        bool keep_right = leaf->is_append_mostly_left();
        node->resize_from_existing(leaf, data_node_type::kMinDensity_, false, keep_left, keep_right, true, stash_frac);
        
        // 3. Update parent node (Need redo upon reocvery)
        std::vector<TraversalNode> traversal_path;
        while(!lock_parent_node(key, &traversal_path, leaf, false)){
          traversal_path.clear();
        }

        resize_log->progress_ = 2; // Inidcator for redo
        my_alloc::BasePMPool::Persist(&(resize_log->progress_), sizeof(resize_log->progress_));

        model_node_type* parent = traversal_path.back().node;
        int bucketID = traversal_path.back().bucketID;
        int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
        int start_bucketID =
          bucketID - (bucketID % repeats);  // first bucket with same child
        int end_bucketID =
          start_bucketID + repeats;  // first bucket with different child
        for (int i = start_bucketID; i < end_bucketID; i++) {
          parent->children_[i] = node;
        }

        my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, sizeof(node) * (end_bucketID - start_bucketID));
        
        // 4. Link to sibling node (Need redo upon reocvery)
        link_resizing_data_nodes(leaf, node);

        node->release_lock();
        parent->release_read_lock();
        safe_delete_node(leaf);

        // 5. clear log
        resize_log->clear_log();
        release_link_locks_for_resizing(node);         
        return true;
      }

      // Not all locks and SMO lock in leaf node are held, remember to release it after the SMO
      std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;
      int fanout_tree_depth = 1;
      if (experimental_params_.splitting_policy_method == 0 || fail >= 2) {
        // always split in 2. No extra work required here
        leaf->build_sorted_slots();
      } else if (experimental_params_.splitting_policy_method == 1) {
        // decide between no split (i.e., expand and retrain) or splitting in
        // 2
        fanout_tree_depth = fanout_tree::find_best_fanout_existing_node_without_parent<T, P>(
            leaf, stats_.num_keys, used_fanout_tree_nodes, 2, true);
      } else if (experimental_params_.splitting_policy_method == 2) {
        // use full fanout tree to decide fanout
        fanout_tree_depth = fanout_tree::find_best_fanout_existing_node_without_parent<T, P>(
            leaf, stats_.num_keys, used_fanout_tree_nodes,
            derived_params_.max_fanout, true);
      }

      if (fanout_tree_depth == 0) {
        // 0. Start logging
        ResizeLog *resize_log = &(my_log.local_log_->resize_log_);
        resize_log->progress_ = 1;
        T insert_key = key;
        memcpy(reinterpret_cast<void*>(resize_log->key_), reinterpret_cast<void*>(&insert_key), sizeof(key));
        resize_log->cur_node_ = pmemobj_oid(reinterpret_cast<void*>(leaf)); // No need to flush; when allocate new node, cur_node_ will also be flushed into the PM

        // 1. Allocate new node
        data_node_type::New_from_existing(&(resize_log->new_node_), leaf);
        auto node = reinterpret_cast<data_node_type*>(pmemobj_direct(resize_log->new_node_));

        // 2. Rehash from old node to new node
        bool keep_left = leaf->is_append_mostly_right();
        bool keep_right = leaf->is_append_mostly_left();
        node->resize_from_existing(leaf, data_node_type::kMinDensity_, true, keep_left, keep_right, false, stash_frac);

        fanout_tree::FTNode& tree_node = used_fanout_tree_nodes[0];
        node->cost_ = tree_node.cost;
        node->expected_avg_search_cost_ =
            tree_node.expected_avg_search_cost;
        node->expected_avg_insert_cost_ = tree_node.expected_avg_insert_cost;
        node->reset_stats();

        
        // 3. Update parent node 
        std::vector<TraversalNode> traversal_path;
        while(!lock_parent_node(key, &traversal_path, leaf, false)){
          traversal_path.clear();
        }

        resize_log->progress_ = 2; // Indicator for redo
        my_alloc::BasePMPool::Persist(&(resize_log->progress_), sizeof(resize_log->progress_));

        model_node_type* parent = traversal_path.back().node;
        int bucketID = traversal_path.back().bucketID;
        int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
        int start_bucketID =
          bucketID - (bucketID % repeats);  // first bucket with same child
        int end_bucketID =
          start_bucketID + repeats;  // first bucket with different child
        for (int i = start_bucketID; i < end_bucketID; i++) {
          parent->children_[i] = node;
        }
        my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, sizeof(node) * (end_bucketID - start_bucketID));

        // 4. Link to sibling node (Need redo upon reocvery)
        link_resizing_data_nodes(leaf, node);

        node->release_lock();
        parent->release_read_lock();
        safe_delete_node(leaf);

        // 5. clear log
        resize_log->clear_log();

        release_link_locks_for_resizing(node);
      } else {
        // split data node: always try to split sideways/upwards, only split
        // downwards if necessary
        bool reuse_model = (fail == 3);
        if (experimental_params_.allow_splitting_upwards) {
          // allow splitting upwards
          // To-DO
        } else {
          // either split sideways or downwards
          split_sideways_downwards_without_parent(leaf, fanout_tree_depth,
                           used_fanout_tree_nodes, reuse_model, key, my_log.local_log_, false, stash_frac);
        }          
      } 
    }

    thread_local int insert_counter(0);
    insert_counter = (insert_counter + 1) & counterMask;
    if(insert_counter == 0){
      ADD(&stats_.num_keys, (1 << 19));
    }

    return true;
  }

 private:
  // Our criteria for when to expand the root, thereby expanding the key domain.
  // We want to strike a balance between expanding too aggressively and too
  // slowly.
  // Specifically, the number of inserted keys falling to the right of the key
  // domain must have one of two properties: (1) above some maximum threshold,
  // or
  // (2) above some minimum threshold and the number is much more than we would
  // expect from randomness alone.
  bool should_expand_right() const {
    return (!root_node_->is_leaf_ &&
            ((istats_.num_keys_above_key_domain >= kMinOutOfDomainKeys &&
              istats_.num_keys_above_key_domain >=
                  kOutOfDomainToleranceFactor *
                      (stats_.num_keys /
                           istats_.num_keys_at_last_right_domain_resize -
                       1)) ||
             istats_.num_keys_above_key_domain >= kMaxOutOfDomainKeys));
  }

  // Similar to should_expand_right, but for insertions to the left of the key
  // domain.
  bool should_expand_left() const {
    return (!root_node_->is_leaf_ &&
            ((istats_.num_keys_below_key_domain >= kMinOutOfDomainKeys &&
              istats_.num_keys_below_key_domain >=
                  kOutOfDomainToleranceFactor *
                      (stats_.num_keys /
                           istats_.num_keys_at_last_left_domain_resize -
                       1)) ||
             istats_.num_keys_below_key_domain >= kMaxOutOfDomainKeys));
  }

  // Expand the key value space that is covered by the index.
  // Expands the root node (which is a model node).
  // If the root node is at the max node size, then we split the root and create
  // a new root node.
  void expand_root(T key, bool expand_left) {
    auto root = static_cast<model_node_type*>(root_node_);
    root_expand_log_->old_root_node_ = pmemobj_oid(root);
    // Find the new bounds of the key domain.
    // Need to be careful to avoid overflows in the key type.
    T domain_size = istats_.key_domain_max_ - istats_.key_domain_min_;
    int expansion_factor;
    T new_domain_min = istats_.key_domain_min_;
    T new_domain_max = istats_.key_domain_max_;
    data_node_type* outermost_node;

    if (expand_left) {
      root_expand_log_->root_expand_in_progress_ = -1;
      auto key_difference = static_cast<double>(istats_.key_domain_min_ -
                                                std::min(key, get_min_key()));
      expansion_factor = pow_2_round_up(static_cast<int>(
          std::ceil((key_difference + domain_size) / domain_size)));
      // Check for overflow. To avoid overflow on signed types while doing
      // this check, we do comparisons using half of the relevant quantities.
      T half_expandable_domain =
          istats_.key_domain_max_ / 2 - std::numeric_limits<T>::lowest() / 2;
      T half_expanded_domain_size = expansion_factor / 2 * domain_size;
      if (half_expandable_domain < half_expanded_domain_size) {
        new_domain_min = std::numeric_limits<T>::lowest();
      } else {
        new_domain_min = istats_.key_domain_max_;
        new_domain_min -= half_expanded_domain_size;
        new_domain_min -= half_expanded_domain_size;
      }
      istats_.num_keys_at_last_left_domain_resize = stats_.num_keys;
      istats_.num_keys_below_key_domain = 0;
      outermost_node = first_data_node();
    } else {
      auto key_difference = static_cast<double>(std::max(key, get_max_key()) -
                                                istats_.key_domain_max_);

      expansion_factor = pow_2_round_up(static_cast<int>(
          std::ceil((key_difference + domain_size) / domain_size)));
      // Check for overflow. To avoid overflow on signed types while doing
      // this check, we do comparisons using half of the relevant quantities.
      T half_expandable_domain =
          std::numeric_limits<T>::max() / 2 - istats_.key_domain_min_ / 2;
      T half_expanded_domain_size = expansion_factor / 2 * domain_size;
      if (half_expandable_domain < half_expanded_domain_size) {
        new_domain_max = std::numeric_limits<T>::max();
      } else {
        new_domain_max = istats_.key_domain_min_;
        new_domain_max += half_expanded_domain_size;
        new_domain_max += half_expanded_domain_size;
      }      
      istats_.num_keys_at_last_right_domain_resize = stats_.num_keys;
      istats_.num_keys_above_key_domain = 0;
      outermost_node = last_data_node();
    }
    assert(expansion_factor > 1);

    // Modify the root node appropriately
    int new_nodes_start;  // index of first pointer to a new node
    int new_nodes_end;    // exclusive
    if (root->num_children_ * expansion_factor <= derived_params_.max_fanout) {
      // During the node expansion, now we still need to replace with the new node
      // Expand root node
      stats_.num_model_node_expansions++;
      stats_.num_model_node_expansion_pointers += root->num_children_;
      int new_num_children = root->num_children_ * expansion_factor;

      PMEMoid tmp;
      model_node_type::New(&tmp, new_num_children);
      auto new_root_node = reinterpret_cast<model_node_type*>(pmemobj_direct(tmp));
      new (new_root_node) model_node_type(*root);
      auto new_children = new_root_node->children_;
      //auto new_children = new (pointer_allocator().allocate(new_num_children))
      //    AlexNode<T, P>*[new_num_children];
      int copy_start;
      if (expand_left) {
        copy_start = new_num_children - root->num_children_;
        new_nodes_start = 0;
        new_nodes_end = copy_start;       
        //root->model_.b_ += new_num_children - root->num_children_;
        new_root_node->model_.b_ += new_num_children - root->num_children_;
      } else {
        copy_start = 0;
        new_nodes_start = root->num_children_;
        new_nodes_end = new_num_children;
      }
      for (int i = 0; i < root->num_children_; i++) {
        new_children[copy_start + i] = root->children_[i];
      }

      //pointer_allocator().deallocate(root->children_, root->num_children_);
      //root->children_ = new_children;
      //root->num_children_ = new_num_children;
      //BT: FIXME, no dellocation of the old root node
      root = new_root_node;
      root_node_ = new_root_node;
      update_superroot_pointer();
    } else {
      // Create new root node
      //auto new_root = new (model_node_allocator().allocate(1))
      //    model_node_type(static_cast<short>(root->level_ - 1), allocator_);
      PMEMoid tmp;
      model_node_type::New(&tmp, expansion_factor);
      auto new_root = reinterpret_cast<model_node_type*>(pmemobj_direct(tmp));
      new (new_root) model_node_type(static_cast<short>(root->level_ - 1), allocator_);
      new_root->model_.a_ = root->model_.a_;
      if (expand_left) {
        new_root->model_.b_ = root->model_.b_ + expansion_factor - 1;
      } else {
        new_root->model_.b_ = root->model_.b_;
      }
      new_root->num_children_ = expansion_factor;
      //new_root->children_ = new (pointer_allocator().allocate(expansion_factor))
      //    AlexNode<T, P>*[expansion_factor];
      if (expand_left) {
        new_root->children_[expansion_factor - 1] = root;
        new_nodes_start = 0;
      } else {
        new_root->children_[0] = root;
        new_nodes_start = 1;
      }
      new_nodes_end = new_nodes_start + expansion_factor - 1;
      root_node_ = new_root;
      update_superroot_pointer();
      root = new_root;
    }
    // Determine if new nodes represent a range outside the key type's domain.
    // This happens when we're preventing overflows.
    int in_bounds_new_nodes_start = new_nodes_start;
    int in_bounds_new_nodes_end = new_nodes_end;
    if (expand_left) {
      in_bounds_new_nodes_start =
          std::max(new_nodes_start, root->model_.predict(new_domain_min));
    } else {
      in_bounds_new_nodes_end =
          std::min(new_nodes_end, root->model_.predict(new_domain_max) + 1);
    }

    // Fill newly created child pointers of the root node with new data nodes.
    // To minimize empty new data nodes, we create a new data node per n child
    // pointers, where n is the number of pointers to existing nodes.
    // Requires reassigning some keys from the outermost pre-existing data node
    // to the new data nodes.
    int n = root->num_children_ - (new_nodes_end - new_nodes_start);
    assert(root->num_children_ % n == 0);

    int new_local_depth = log_2_round_down(root->num_children_) - log_2_round_down(n);
    if (expand_left) {
      T left_boundary_value = istats_.key_domain_min_;
      int left_boundary = outermost_node->lower_bound(left_boundary_value, true, true);
      data_node_type* next = outermost_node;
      for (int i = new_nodes_end; i > new_nodes_start; i -= n) {
        if (i <= in_bounds_new_nodes_start) {
          // Do not initialize nodes that fall outside the key type's domain
          break;
        }
        int right_boundary = left_boundary;
        if (i - n <= in_bounds_new_nodes_start) {
          left_boundary = 0;
        } else {
          left_boundary_value -= domain_size;
          left_boundary = outermost_node->lower_bound(left_boundary_value, true);
        }
        data_node_type* new_node = bulk_load_leaf_node_from_existing(
            outermost_node, left_boundary, right_boundary, true, nullptr, false, false,false);
        new_node->level_ = static_cast<short>(root->level_ + 1);
        new_node->local_depth_ = new_local_depth;
        if (next) {
          next->prev_leaf_ = new_node;
        }
        new_node->next_leaf_ = next;
        next = new_node;
        
        for (int j = i - 1; j >= i - n; j--) {
          root->children_[j] = new_node;
        }
      }
    } else {
      T right_boundary_value = istats_.key_domain_max_;
      int right_boundary = outermost_node->lower_bound(right_boundary_value, true, true);
      data_node_type* prev = nullptr;
      for (int i = new_nodes_start; i < new_nodes_end; i += n) {
        if (i >= in_bounds_new_nodes_end) {
          // Do not initialize nodes that fall outside the key type's domain
          break;
        }
        int left_boundary = right_boundary;
        if (i + n >= in_bounds_new_nodes_end) {
          right_boundary = outermost_node->data_capacity_;
        } else {
          right_boundary_value += domain_size;
          right_boundary = outermost_node->lower_bound(right_boundary_value, true);
        }
        data_node_type* new_node = bulk_load_leaf_node_from_existing(
            outermost_node, left_boundary, right_boundary, true, nullptr, false, false,false);
        new_node->level_ = static_cast<short>(root->level_ + 1);
        new_node->local_depth_ = new_local_depth;
        if (prev) {
          prev->next_leaf_ = new_node;
        }
        new_node->prev_leaf_ = prev;
        prev = new_node;
        for (int j = i; j < i + n; j++) {
          root->children_[j] = new_node;
        }
      }
    }

    // Now the log is ready, we could enters the redo phase

    // Connect leaf nodes and remove reassigned keys from outermost pre-existing
    // node.
    if (expand_left) {
      outermost_node->erase_range(root_expand_log_->new_domain_min_, istats_.key_domain_min_);
      auto last_new_leaf =
          static_cast<data_node_type*>(root->children_[new_nodes_end - 1]);
      outermost_node->prev_leaf_ = last_new_leaf;
      last_new_leaf->next_leaf_ = outermost_node;
    } else {
      outermost_node->erase_range(istats_.key_domain_max_, new_domain_max,
                                  true);
      auto first_new_leaf =
          static_cast<data_node_type*>(root->children_[new_nodes_start]);
      outermost_node->next_leaf_ = first_new_leaf;
      first_new_leaf->prev_leaf_ = outermost_node;
    }

    istats_.key_domain_min_ = new_domain_min;
    istats_.key_domain_max_ = new_domain_max;
  }

  // Splits downwards in the manner determined by the fanout tree and updates
  // the pointers of the parent.
  // If no fanout tree is provided, then splits downward in two. Returns the
  // newly created model node.
  void split_sideways_downwards_without_parent(
      data_node_type* leaf, int fanout_tree_depth,
      std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
      bool reuse_model, T key, MyLog *my_log, bool build_sorted_node = false, double stash_frac = 0.05) {
     data_node_type* left_leaf = nullptr;
     data_node_type* right_leaf = nullptr;

     // 0. Start logging
     auto log = &(my_log->smo_log_);
     log->cur_node_ = pmemobj_oid(leaf);
     memcpy(reinterpret_cast<void*>(log->key_), reinterpret_cast<void*>(&key), sizeof(T));
     log->progress_ = 1;
     log->fanout_tree_depth = fanout_tree_depth;
     my_alloc::BasePMPool::Persist(&log->cur_node_, 28);

     // 1. Allocate new nodes, split the records from old node to new node, and update the side links with brother nodes
    if (used_fanout_tree_nodes.empty()) {
        assert(fanout_tree_depth == 1);
        create_two_new_data_nodes_without_parent(leaf, reuse_model, log->reserved_child_, log->reserved_child_ + 1, build_sorted_node, stash_frac);
        left_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(log->reserved_child_[0]));
        right_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(log->reserved_child_[1]));
    } else {
        create_new_data_nodes_without_parent(leaf, used_fanout_tree_nodes, log, build_sorted_node, stash_frac);
    }

RETRAVEL:
    
    // 2. Ready for lock and Update the parent
    std::vector<TraversalNode> traversal_path;
    while(!lock_parent_node(key, &traversal_path, leaf, false)){
      traversal_path.clear();
    }
    model_node_type* parent = traversal_path.back().node; 
    if (parent == superroot_) {
      update_superroot_key_domain();
    }
    int bucketID = parent->model_.predict(key);
    bucketID = std::min<int>(std::max<int>(bucketID, 0),
                             parent->num_children_ - 1);

    int fanout = 1 << fanout_tree_depth;
    int leaf_duplication = log_2_round_down(parent->num_children_) - leaf->local_depth_;
    bool should_split_downwards =
            (parent->num_children_ * fanout /
                     (1 << leaf_duplication) >
                 derived_params_.max_fanout ||
             parent->level_ == superroot_->level_);    

    if (should_split_downwards)
    { 
      // 3. Split downwards
      log->progress_ = 2;
      my_alloc::BasePMPool::Persist(&log->progress_, sizeof(log->progress_));
      // Parent node size > max. node size
      model_node_type::New(&(log->downward_node), fanout);
      auto new_node = reinterpret_cast<model_node_type*>(pmemobj_direct(log->downward_node));
      new (new_node) model_node_type(leaf->level_, allocator_);
      new_node->local_depth_ = leaf->local_depth_;
      new_node->num_children_ = fanout;
      new_node->model_.a_ =
          1.0 / (leaf->max_limit_ - leaf->min_limit_) * fanout;
      new_node->model_.b_ = -new_node->model_.a_ * leaf->min_limit_;

      // since the parent has been locked, we start updating some critical info
      if (used_fanout_tree_nodes.empty()) {
        left_leaf->level_ = static_cast<short>(new_node->level_ + 1);
        right_leaf->level_ = static_cast<short>(new_node->level_ + 1);
        left_leaf->local_depth_ = 1;
        right_leaf->local_depth_ = 1;
        my_alloc::BasePMPool::Persist(&(left_leaf->local_depth_), sizeof(left_leaf->local_depth_) * 2);
        my_alloc::BasePMPool::Persist(&(right_leaf->local_depth_), sizeof(right_leaf->local_depth_) * 2);
        left_leaf->min_limit_ = (0 - new_node->model_.b_) / new_node->model_.a_;
        left_leaf->max_limit_ = (1 - new_node->model_.b_) / new_node->model_.a_;
        right_leaf->min_limit_ = left_leaf->max_limit_;
        right_leaf->max_limit_ = (2 - new_node->model_.b_) / new_node->model_.a_;
        my_alloc::BasePMPool::Persist(&(left_leaf->min_limit_), sizeof(left_leaf->min_limit_) * 2);
        my_alloc::BasePMPool::Persist(&(right_leaf->min_limit_), sizeof(right_leaf->min_limit_) * 2);
        new_node->children_[0] = left_leaf;
        new_node->children_[1] = right_leaf;
      }else{
        left_leaf = nullptr;
        right_leaf = nullptr;      
        int cur = 0;
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          data_node_type* child_node = reinterpret_cast<data_node_type*>(tree_node.data_node);
          child_node->level_ = static_cast<short>(new_node->level_ + 1);
          child_node->local_depth_ = tree_node.level;
          my_alloc::BasePMPool::Persist(&(child_node->local_depth_), sizeof(child_node->local_depth_) * 2);
          int child_node_repeats = 1 << (fanout_tree_depth - child_node->local_depth_);
          child_node->min_limit_ = (cur - new_node->model_.b_) / new_node->model_.a_;
          child_node->max_limit_ = (cur + child_node_repeats - new_node->model_.b_) / new_node->model_.a_;
          my_alloc::BasePMPool::Persist(&(child_node->max_limit_), sizeof(child_node->max_limit_) * 2);
          for(int i = cur; i < cur + child_node_repeats; ++i){
            new_node->children_[i] = child_node;
          }
          cur += child_node_repeats;
          if(left_leaf == nullptr){
            left_leaf = child_node;
          }
          right_leaf = child_node;
        }
      }

      new_node->get_read_lock();
      my_alloc::BasePMPool::Persist(new_node, new_node->get_node_size());

      log->progress_ = 3;
      my_alloc::BasePMPool::Persist(&log->progress_, sizeof(log->progress_)); // Undo/Redo dividing line
      
      // update the pointers in parent
      int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
      int start_bucketID =
          bucketID - (bucketID % repeats);  // first bucket with same child
      int end_bucketID =
          start_bucketID + repeats;  // first bucket with different child
      for (int i = start_bucketID; i < end_bucketID; i++) {
        parent->children_[i] = new_node;
      }
      my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, sizeof(new_node) * (end_bucketID - start_bucketID));

      link_data_nodes(leaf, left_leaf, right_leaf);
      if (parent == superroot_) {
        root_node_ = new_node;
        my_alloc::BasePMPool::Persist(&root_node_, sizeof(root_node_));
        update_superroot_pointer();
      }

      safe_delete_node(leaf);
      new_node->release_read_lock();
      parent->release_read_lock();
      log->clear_log();
      release_link_locks_for_split(left_leaf, right_leaf);
      // Release the locks for all data nodes

      if (used_fanout_tree_nodes.empty()) {
        left_leaf->release_lock();
        right_leaf->release_lock();
      }else{
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          data_node_type* child_node = reinterpret_cast<data_node_type*>(tree_node.data_node);
          child_node->release_lock();
        }
      }
    }else{
      int compute_duplication = log_2_round_down(parent->num_children_) - leaf->local_depth_;
      int repeats = 1 << compute_duplication;
      bool parent_write_lock = false;
      if (fanout > repeats)
      {
        if (!parent->promote_from_read_to_write())
        {
          parent->release_read_lock();
          goto RETRAVEL;
        }

        // 4. Resize the parent
        log->progress_ = 4;
        my_alloc::BasePMPool::Persist(&log->progress_, sizeof(log->progress_));

        auto resize_log = &(log->model_resize_log_);
        resize_log->cur_node_ = pmemobj_oid(parent);
        parent_write_lock = true;
        int expansion_factor = 1 << (fanout_tree_depth - compute_duplication);
        model_node_type::expand(&(resize_log->new_node_), parent, fanout_tree_depth - compute_duplication);
        auto new_node = reinterpret_cast<model_node_type*>(pmemobj_direct(resize_log->new_node_));

        // If the new node has been allocated and initialized, then redo
        // Lock parent node and update the pointer
        std::vector<TraversalNode> traversal_path;
        while(!lock_parent_node(key, &traversal_path, parent)){
          traversal_path.clear();
        }

        model_node_type *grand_parent = traversal_path.back().node;
        auto parent_bucketID = traversal_path.back().bucketID;
        int parent_repeats = 1 << (log_2_round_down(grand_parent->num_children_) - parent->local_depth_);
        int parent_start_bucketID = parent_bucketID - (parent_bucketID % parent_repeats);
        int parent_end_bucketID = parent_start_bucketID + parent_repeats;
        for (int i = parent_start_bucketID; i < parent_end_bucketID; ++i)
        {
          grand_parent->children_[i] = new_node;
        }
        my_alloc::BasePMPool::Persist(grand_parent->children_ + parent_start_bucketID, parent_repeats * sizeof(new_node));

        //if grand_parent is the root node, need to update it
        if(grand_parent == superroot_){
          root_node_ = new_node;
          my_alloc::BasePMPool::Persist(&root_node_, sizeof(new_node));
          update_superroot_pointer();
        }
        grand_parent->release_read_lock();
        auto old_parent = parent;
        old_parent->is_obsolete_ = true;
        safe_delete_node(old_parent);
        
         // 5. Finish the model expansion
        log->progress_ = 1;
        my_alloc::BasePMPool::Persist(&log->progress_, sizeof(log->progress_));

        parent = new_node;
        repeats *= expansion_factor;
        bucketID *= expansion_factor;
      }

      int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child
      int end_bucketID =
        start_bucketID + repeats;  // first bucket with different child
      if (used_fanout_tree_nodes.empty()) { // Now only focus on split into two nodes
        assert(fanout_tree_depth == 1);
        int mid_bucketID = start_bucketID + repeats / 2;
        left_leaf->local_depth_ = leaf->local_depth_ + 1;
        right_leaf->local_depth_ = leaf->local_depth_ + 1;
        left_leaf->level_ = static_cast<short>(parent->level_ + 1);
        right_leaf->level_ = static_cast<short>(parent->level_ + 1);
        my_alloc::BasePMPool::Persist(&(left_leaf->local_depth_), sizeof(left_leaf->local_depth_) * 2);
        my_alloc::BasePMPool::Persist(&(right_leaf->local_depth_), sizeof(right_leaf->local_depth_) * 2);
        left_leaf->min_limit_ = (start_bucketID - parent->model_.b_) / parent->model_.a_;
        left_leaf->max_limit_ = (mid_bucketID - parent->model_.b_) / parent->model_.a_;
        right_leaf->min_limit_ = left_leaf->max_limit_;
        right_leaf->max_limit_ = (end_bucketID - parent->model_.b_) / parent->model_.a_;
        my_alloc::BasePMPool::Persist(&left_leaf->max_limit_, sizeof(double) * 2);
        my_alloc::BasePMPool::Persist(&right_leaf->max_limit_, sizeof(double) * 2);

        log->progress_ = 5; // Redo occurs below
        my_alloc::BasePMPool::Persist(&log->progress_, sizeof(log->progress_));

        for (int i = start_bucketID; i < mid_bucketID; i++) {
          parent->children_[i] = left_leaf;
        }

        for (int i = mid_bucketID; i < end_bucketID; i++) {
          parent->children_[i] = right_leaf;
        }
      } else {
        // Extra duplication factor is required when there are more redundant
        // pointers than necessary
        left_leaf = nullptr;
        right_leaf = nullptr;
        int cur = start_bucketID;
        int global_depth = log_2_round_down(parent->num_children_);
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          data_node_type* child_node = reinterpret_cast<data_node_type*>(tree_node.data_node);
          child_node->level_ = static_cast<short>(parent->level_ + 1);
          child_node->local_depth_ = leaf->local_depth_ + tree_node.level;
          my_alloc::BasePMPool::Persist(&(child_node->local_depth_), sizeof(child_node->local_depth_) * 2);
          int child_node_repeats = 1 << (global_depth - child_node->local_depth_);
          child_node->min_limit_ = (cur - parent->model_.b_) / parent->model_.a_;
          child_node->max_limit_ = (cur + child_node_repeats - parent->model_.b_) / parent->model_.a_;
          my_alloc::BasePMPool::Persist(&child_node->max_limit_, sizeof(double) * 2);
          cur += child_node_repeats;
          if(left_leaf == nullptr){
            left_leaf = child_node;
          }
          right_leaf = child_node;
        }

        log->progress_ = 5; // Redo occurs below
        my_alloc::BasePMPool::Persist(&log->progress_, sizeof(log->progress_));

        cur = start_bucketID;
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          data_node_type* child_node = reinterpret_cast<data_node_type*>(tree_node.data_node);
          int child_node_repeats = 1 << (global_depth - child_node->local_depth_);
          for(int i = cur; i < cur + child_node_repeats; ++i){
            parent->children_[i] = child_node;
          }
          cur += child_node_repeats;
        }
      }
      my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, repeats);
      link_data_nodes(leaf, left_leaf, right_leaf);
      safe_delete_node(leaf);
      if(parent_write_lock){
        parent->release_write_lock();
      }else{
        parent->release_read_lock();
      }
      log->clear_log();
      release_link_locks_for_split(left_leaf, right_leaf);

      if (used_fanout_tree_nodes.empty()) {
        left_leaf->release_lock();
        right_leaf->release_lock();
      }else{
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          data_node_type* child_node = reinterpret_cast<data_node_type*>(tree_node.data_node);
          child_node->release_lock();
        }
      }
    }
  }

void create_two_new_data_nodes_without_parent(data_node_type* old_node, bool reuse_model, PMEMoid *split_left, PMEMoid *split_right, bool build_sorted_node = false, double stash_frac = 0.05) {
    double mid_limit = old_node->min_limit_ + (old_node->max_limit_ - old_node->min_limit_) / 2.0;
    bool append_mostly_right = old_node->is_append_mostly_right();
    bool appending_right_fit = old_node->max_key_ < mid_limit ? true : false;   
    bool append_mostly_left = old_node->is_append_mostly_left();
    bool appending_left_fit = old_node->min_key_ < mid_limit ? true : false;

    int right_boundary = old_node->lower_bound(mid_limit, true);    

    data_node_type* left_leaf = bulk_load_leaf_node_from_existing(
        old_node, 0, right_boundary, true, nullptr, reuse_model,
        append_mostly_right && appending_right_fit,
        append_mostly_left && appending_left_fit , build_sorted_node, split_left, stash_frac, true);
    // set the limit of the left leaf 
    //FIXME: no need here, should set limit according to parent  
    left_leaf->min_limit_ = old_node->min_limit_;
    left_leaf->max_limit_ = mid_limit;

    appending_right_fit = old_node->max_key_ >= mid_limit ? true : false;
    appending_left_fit = old_node->min_key_ >= mid_limit ? true : false;

    data_node_type* right_leaf = bulk_load_leaf_node_from_existing(
        old_node, right_boundary, old_node->num_keys_, true, nullptr,
        reuse_model,
        append_mostly_right && appending_right_fit,
        append_mostly_left && appending_left_fit, build_sorted_node, split_right, stash_frac, false);
    right_leaf->min_limit_ = left_leaf->max_limit_;
    right_leaf->max_limit_ = old_node->max_limit_;

    //Inidcate that this node has been deleted, should be carefully considerred when linking nodes
    old_node->is_obsolete_ = true;
    left_leaf->next_leaf_ = right_leaf; // thread-safe update
    right_leaf->prev_leaf_ = left_leaf; // thread-safe update
    left_leaf->lock_ = lockSet; // Set the lock
    right_leaf->lock_ = lockSet;
  }

  void validate_node_split(data_node_type* old_node, data_node_type* left_leaf, data_node_type* right_leaf){
    std::cout << "Start output the structure of old node" << std::endl;
    old_node->OutputDatanodeInfo();
    std::cout << "Start output the structure of left node" << std::endl;
    left_leaf->OutputDatanodeInfo();
    std::cout << "Start output the structure of right node" << std::endl;
    right_leaf->OutputDatanodeInfo();
  }


void create_new_data_nodes_without_parent(
      data_node_type* old_node,
      std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes, SplitSidewayDownwardsLog* log, bool build_sorted_node = false, double stash_frac = 0.05) {
    bool append_mostly_right = old_node->is_append_mostly_right();
    bool append_mostly_left = old_node->is_append_mostly_left();
    old_node->is_obsolete_ = true;
    // Create the new data nodes
    data_node_type* prev_leaf = nullptr;  // used for linking the new data nodes
    auto split_num = used_fanout_tree_nodes.size();
    PMEMoid* child_nodes;
    int *tree_level = nullptr;

    if(split_num <= 2){
      child_nodes = log->reserved_child_;
    }else{
      log->split_num_ = split_num;
      my_alloc::BasePMPool::Persist(&(log->split_num_), sizeof(int));
      my_alloc::BasePMPool::ZAllocate(&(log->child_nodes_), (sizeof(PMEMoid) + sizeof(int)) * split_num);
      child_nodes = reinterpret_cast<PMEMoid*>(pmemobj_direct(log->child_nodes_));
      tree_level = reinterpret_cast<int*>(child_nodes + split_num);
    }
    int j = 0;

    for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
      bool keep_left = append_mostly_right && tree_node.left_limit <= old_node->max_key_ && old_node->max_key_ < tree_node.right_limit;
      bool keep_right = append_mostly_left && tree_node.left_limit <= old_node->min_key_ && old_node->min_key_ < tree_node.right_limit;

      if(tree_level != nullptr){
        tree_level[j] = tree_node.level;
      }
      data_node_type* child_node = bulk_load_leaf_node_from_existing(
          old_node, tree_node.left_boundary, tree_node.right_boundary, false,
          &tree_node, false, keep_left, keep_right, build_sorted_node, child_nodes + j, stash_frac);
      j++;
      //child_node->level_ = static_cast<short>(parent->level_ + 1);
      child_node->lock_ = lockSet;
      child_node->cost_ = tree_node.cost;
#ifdef NEW_COST_MODEL
      child_node->expected_avg_search_cost_ =
          tree_node.expected_avg_search_cost;
      child_node->expected_avg_insert_cost_ = tree_node.expected_avg_insert_cost;
#else
      child_node->expected_avg_exp_search_iterations_ =
          tree_node.expected_avg_search_iterations;
      child_node->expected_avg_shifts_ = tree_node.expected_avg_shifts;
#endif
      tree_node.data_node = reinterpret_cast<void*>(child_node);
    
      if(prev_leaf != nullptr){
        prev_leaf->next_leaf_ = child_node;
        child_node->prev_leaf_ = prev_leaf;
      }
      prev_leaf = child_node;
    }

    if (tree_level != nullptr)
    {
      // Persist the tree level information
      my_alloc::BasePMPool::Persist(tree_level, split_num * sizeof(int));
    }
  }


  /*** Delete, Update, Range Query ***/

 public:

  // Erases all keys with a certain key value
  // Return the number of keys erased: 0 or 1 in primary index
  // This function now has concurrency support
  bool erase(const T& key, bool epoch = false){
    if(epoch){
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      return erase_unsafe(key);
    }
    return erase_unsafe(key);
  }

  bool erase_unsafe(const T& key) {
    int num_erased = 0;
    do{
      data_node_type* leaf = get_leaf(key);
      // First test it whether it needs to be recoverred
      if(leaf->local_version_ != global_version_){
        leaf->recover_node(global_version_);
      }
      int ret = leaf->erase(key, &num_erased);
      if(ret == 0){
        // Delete success, means no concurrency conflict
        // Delete 0 or 1 item
        break;
      }
    }while(true);
    
    thread_local int erase_counter(1);
    erase_counter = (erase_counter + num_erased) & counterMask;
    if(erase_counter == 0){
      SUB(&stats_.num_keys, (1 << 19) - 1);
      erase_counter = 1;
    }

    if (key > istats_.key_domain_max_) {
      istats_.num_keys_above_key_domain -= num_erased;
    } else if (key < istats_.key_domain_min_) {
      istats_.num_keys_below_key_domain -= num_erased;
    }

    return num_erased;
  }

  bool update(const T& key, const P& payload, bool epoch = false){
    if(epoch){
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      return update_unsafe(key, payload);
    }
    return update_unsafe(key, payload);
  }

  bool update_unsafe(const T& key, const P& payload) {
    int num_updated = 0;
    do{
      data_node_type* leaf = get_leaf(key);
      // First test it whether it needs to be recoverred
      if(leaf->local_version_ != global_version_){
        leaf->recover_node(global_version_);
      }
      int ret = leaf->update(key, payload, &num_updated);
      if(ret == 0){
        // Update success, means no concurrency conflict
        // Update 0 or 1 item
        break;
      }
    }while(true);

    return num_updated;
  }

  int range_scan_by_size(const T& key, uint32_t to_scan, V* &result = nullptr, bool epoch = false){
    if(epoch){
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      return range_scan_by_size_unsafe(key, to_scan, result);
    }
    return range_scan_by_size_unsafe(key, to_scan, result);
  }

  // Return the actual scan size
  // And put the sorted results in result array
  // The application could provide the result array to avoid the overhead of memory overhead
  int range_scan_by_size_unsafe(const T& key, uint32_t to_scan, V* &result){
    if (result == nullptr)
    {
      // If the application does not provide result array, index itself creates the returned storage
      result = new V[to_scan];
    }

    data_node_type* leaf = get_leaf(key);
    // During scan, needs to guarantee the atomic read of each record (Optimistic CC)
    return leaf->range_scan_by_size(key, to_scan, result, global_version_);
  }

  // Return the scan size, the return results should >= key1 && < key2
  int range_scan_by_key(const T& key1, const T& key2, V* result){
    // TO-DO
    return 0;
  }


  /* Recovery */

  // Redo data node resizing
  inline void recover_from_resize(data_node_type* cur_node, data_node_type *new_node, T key){
    // 1. Re-update the pointers in parent
    std::vector<TraversalNode> traversal_path;
    get_leaf_with_traversal(key, &traversal_path);
    model_node_type *parent = traversal_path.back().node;
    int bucketID = traversal_path.back().bucketID;
    int repeats = 1 << (log_2_round_down(parent->num_children_) - cur_node->local_depth_);
    int start_bucketID =
      bucketID - (bucketID % repeats);  // first bucket with same child
    int end_bucketID =
      start_bucketID + repeats;  // first bucket with different child
    for (int i = start_bucketID; i < end_bucketID; i++) {
      parent->children_[i] = new_node;
    }
    my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, sizeof(new_node) * (end_bucketID - start_bucketID));

    // 2. Link sibling nodes
    link_resizing_data_nodes_without_lock(cur_node, new_node);

    // 3. Release locked locks
    uint32_t lock_version;
    if(new_node->test_lock_set(lock_version)){
      new_node->release_lock();
    }
    parent->reset_rw_lock();
    release_link_locks_for_resizing(new_node);
  }

  inline void undo_smo_log(SplitSidewayDownwardsLog *smo_log){
    if(!OID_IS_NULL(smo_log->reserved_child_[0])){
      auto new_node = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->reserved_child_[0]));
      new_node->recover_reclaim();
      pmemobj_free(&(smo_log->reserved_child_[0]));
      if (!OID_IS_NULL(smo_log->reserved_child_[1]))
      {
        new_node = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->reserved_child_[1]));
        new_node->recover_reclaim();
        pmemobj_free(&(smo_log->reserved_child_[1]));
      }
    }else if(!OID_IS_NULL(smo_log->child_nodes_)){
      // Reclaim multiple nodes
      PMEMoid *child_nodes = reinterpret_cast<PMEMoid*>(pmemobj_direct(smo_log->child_nodes_));
      auto child_num = smo_log->split_num_;
      for(int i = 0; i < child_num; ++i){
        if(!OID_IS_NULL(child_nodes[i])){
          auto new_node = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[i]));
          new_node->recover_reclaim();
          pmemobj_free(&(child_nodes[i]));
        }
      }
      pmemobj_free(&(smo_log->child_nodes_));
    }
  }

  inline void redo_split_downwards(SplitSidewayDownwardsLog *smo_log){ 
    T key;
    memcpy(&key, smo_log->key_, sizeof(key)); 
    data_node_type *leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->cur_node_));
    model_node_type *new_node = reinterpret_cast<model_node_type*>(pmemobj_direct(smo_log->downward_node));

    // 1. Update the parent
    std::vector<TraversalNode> traversal_path;
    get_leaf_with_traversal(key, &traversal_path);
    model_node_type *parent = traversal_path.back().node;
    int bucketID = traversal_path.back().bucketID;
    int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
    int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child
    int end_bucketID =
        start_bucketID + repeats;  // first bucket with different child
    for (int i = start_bucketID; i < end_bucketID; i++) {
      parent->children_[i] = new_node;
    }
    my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, sizeof(new_node) * (end_bucketID - start_bucketID));
    
    // 2. Link the sibling nodes
    data_node_type *left_leaf = nullptr;
    data_node_type *right_leaf = nullptr;
    if(!OID_IS_NULL(smo_log->reserved_child_[0])){
      left_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->reserved_child_[0]));
      right_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->reserved_child_[1]));
      left_leaf->release_lock();
      right_leaf->release_lock();
    }else{
      int split_num = smo_log->split_num_;
      PMEMoid *child_nodes = reinterpret_cast<PMEMoid*>(pmemobj_direct(smo_log->child_nodes_));
      left_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[0]));
      right_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[split_num-1]));
      for(int i = 0; i < split_num; ++i){
        auto cur_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[i]));
        cur_leaf->release_lock();
      }
    }
    link_data_nodes_without_lock(leaf, left_leaf, right_leaf);

    // 3. Reset locked states
    new_node->reset_rw_lock();
    parent->reset_rw_lock();
    release_link_locks_for_split(left_leaf, right_leaf);
    pmemobj_free(&(smo_log->cur_node_));
  }

  inline void redo_model_resizing(NewModelResizeLog *model_log, T key){ 
    if((OID_IS_NULL(model_log->cur_node_)) || (OID_IS_NULL(model_log->new_node_))){ 
      return;
    }
    auto parent = reinterpret_cast<model_node_type*>(pmemobj_direct(model_log->cur_node_));
    auto new_node = reinterpret_cast<model_node_type*>(pmemobj_direct(model_log->new_node_));
    std::vector<TraversalNode> traversal_path;
    get_leaf_with_traversal(key, &traversal_path);
    traversal_path.pop_back();

    model_node_type *grand_parent = traversal_path.back().node;
    auto parent_bucketID = traversal_path.back().bucketID;
    int parent_repeats = 1 << (log_2_round_down(grand_parent->num_children_) - parent->local_depth_);
    int parent_start_bucketID = parent_bucketID - (parent_bucketID % parent_repeats);
    int parent_end_bucketID = parent_start_bucketID + parent_repeats;
    for (int i = parent_start_bucketID; i < parent_end_bucketID; ++i)
    {
      grand_parent->children_[i] = new_node;
    }
    my_alloc::BasePMPool::Persist(grand_parent->children_ + parent_start_bucketID, parent_repeats * sizeof(new_node));

    //if grand_parent is the root node, need to update it
    if(grand_parent == superroot_){
      root_node_ = new_node;
      my_alloc::BasePMPool::Persist(&root_node_, sizeof(new_node));
      update_superroot_pointer();
    }

    grand_parent->reset_rw_lock();
    new_node->reset_rw_lock();
    pmemobj_free(&(model_log->cur_node_));
  }

  // Update the parent and link resizing ndoes
  inline void redo_smo_log(SplitSidewayDownwardsLog *smo_log){
    T key;
    memcpy(&key, smo_log->key_, sizeof(key)); 
    data_node_type *leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->cur_node_));

    // Update the parent
    std::vector<TraversalNode> traversal_path;
    get_leaf_with_traversal(key, &traversal_path);
    model_node_type *parent = traversal_path.back().node;
    int bucketID = traversal_path.back().bucketID;
    int repeats = 1 << (log_2_round_down(parent->num_children_) - leaf->local_depth_);
    int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child
    int end_bucketID =
        start_bucketID + repeats;  // first bucket with different child
    data_node_type *left_leaf = nullptr;
    data_node_type *right_leaf = nullptr;
    if(!OID_IS_NULL(smo_log->reserved_child_[0])){
      left_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->reserved_child_[0]));
      right_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(smo_log->reserved_child_[1]));
      int mid_bucketID = start_bucketID + repeats / 2;

      for (int i = start_bucketID; i < mid_bucketID; i++) {
        parent->children_[i] = left_leaf;
      }
      for (int i = mid_bucketID; i < end_bucketID; i++) {
        parent->children_[i] = right_leaf;
      }

      left_leaf->release_lock();
      right_leaf->release_lock();
    }else{
      int split_num = smo_log->split_num_;
      PMEMoid *child_nodes = reinterpret_cast<PMEMoid*>(pmemobj_direct(smo_log->child_nodes_));
      left_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[0]));
      right_leaf = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[split_num-1]));
      int global_depth = log_2_round_down(parent->num_children_);
      int cur = start_bucketID;

      for(int i = 0; i < split_num; i++){
        data_node_type* child_node = reinterpret_cast<data_node_type*>(pmemobj_direct(child_nodes[i]));
        child_node->release_lock();
        int child_node_repeats = 1 << (global_depth - child_node->local_depth_);
        for(int i = cur; i < cur + child_node_repeats; ++i){
          parent->children_[i] = child_node;
        }
        cur += child_node_repeats;
      }
    }
    my_alloc::BasePMPool::Persist(parent->children_ + start_bucketID, repeats);

    // 2. link sibling nodes
    link_data_nodes_without_lock(leaf, left_leaf, right_leaf);
    
    // 3. clear locks
    parent->reset_rw_lock();
    release_link_locks_for_split(left_leaf, right_leaf);
    
    pmemobj_free(&(smo_log->cur_node_));
  }

  void recover_from_log(MyLog *log){
    // OverflowDesc recovery
    OverflowDesc *overflow_desc = &(log->overflow_desc_);
    if((overflow_desc->cur_node_ != nullptr) && (!OID_IS_NULL(overflow_desc->new_node_))){
      // Undo logging
      auto cur_data_node = reinterpret_cast<data_node_type*>(overflow_desc->cur_node_);
      auto overflow_stash = reinterpret_cast<typename data_node_type::overflow_stash_type*>(pmemobj_direct(overflow_desc->new_node_));
      if(!cur_data_node->is_overflow_stash_linked(overflow_stash)){
        pmemobj_free(&(overflow_desc->new_node_));
      }
      overflow_desc->cur_node_ = nullptr;
      my_alloc::BasePMPool::Persist(&(overflow_desc->cur_node_), sizeof(overflow_desc->cur_node_));
    }

    // Data Node resize recovery
    auto resize_log = &(log->resize_log_);
    if (resize_log->progress_ == 1)
    {
      // Undo
      if(!OID_IS_NULL(resize_log->new_node_)){
        // safe destory
        auto new_node = reinterpret_cast<data_node_type*>(pmemobj_direct(resize_log->new_node_));
        new_node->recover_reclaim();
        pmemobj_free(&(resize_log->new_node_));
      }
      resize_log->clear_log();
    }else if (resize_log->progress_ == 2){
      // Redo
      if(!OID_IS_NULL(resize_log->cur_node_)){
        auto new_node = reinterpret_cast<data_node_type*>(pmemobj_direct(resize_log->new_node_));
        auto cur_node = reinterpret_cast<data_node_type*>(pmemobj_direct(resize_log->cur_node_));
        T key;
        memcpy(&key, resize_log->key_, sizeof(key)); 
        recover_from_resize(cur_node, new_node, key);
        cur_node->recover_reclaim();
        pmemobj_free(&(resize_log->cur_node_));
      }
      resize_log->clear_log();
    }

    // Data Node SMO recovery
    SplitSidewayDownwardsLog *smo_log = &(log->smo_log_);
    if(smo_log->progress_ > 0){
      switch(smo_log->progress_){
        case 1 : // Undo
        {
          undo_smo_log(smo_log);
        }
        break;

        case 2 : // undo split downwards
        {
          if(!OID_IS_NULL(smo_log->downward_node)){
            pmemobj_free(&(smo_log->downward_node));
          }
          undo_smo_log(smo_log);
        }
        break;

        case 3 :  // redo split downwards
        {
          redo_split_downwards(smo_log);
        }
        break;

        case 4 : // redo the model resizing but undo the data node split
        {
          auto model_resize_log = &(smo_log->model_resize_log_);
          T key;
          memcpy(&key, smo_log->key_, sizeof(key)); 
          redo_model_resizing(model_resize_log, key);
          undo_smo_log(smo_log);
        }
        break;

        case 5 : // redo the node split!
        {
          redo_smo_log(smo_log);
        }
      }

      smo_log->clear_log();
    }

    log->in_use = 0;
    my_alloc::BasePMPool::Persist(&(log->in_use), sizeof(log->in_use));
  }

  void recovery(){ 
    // Epoch recovery
    my_alloc::BasePMPool::EpochRecovery();

    // Release lock in root
    release_lock();

    global_version_++;
    my_alloc::BasePMPool::Persist(&global_version_, sizeof(global_version_));
    if(global_version_ == 0){
      // Need reset all data nodes
      for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
        AlexNode<T, P>* cur = node_it.current();
        if (cur->is_leaf_) {
          AlexDataNode<T, P>* cur_data_node = reinterpret_cast<AlexDataNode<T, P>*>(cur);
          cur_data_node->local_version_ = 1;
        }
      }
    }
    // Check logging
    int log_size = log_->log_size();
    for(int i = 0; i < log_size; ++i){
      MyLog* cur_log = log_->log_array_ + i;
      if (cur_log->in_use)
      {
        recover_from_log(cur_log);
      }else{
        break;
      }
    }
  }

 private:
  // Try to merge empty leaf, which can be traversed to by looking up key
  // This may cause the parent node to merge up into its own parent
  void merge(data_node_type* leaf, T key) {
    std::cout << "FATAL: merge is not implemented in this version" << std::endl;
    exit(-1);
  }

  /*** Stats ***/

 public:
  // Number of elements
  size_t size() const { return static_cast<size_t>(stats_.num_keys); }

  // Iterates through all nodes with pre-order traversal
  class NodeIterator {
   public:
    const self_type* index_;
    AlexNode<T, P>* cur_node_;
    std::stack<AlexNode<T, P>*> node_stack_;  // helps with traversal

    // Start with root as cur and all children of root in stack
    explicit NodeIterator(const self_type* index)
        : index_(index), cur_node_(index->root_node_) {
      if (cur_node_ && !cur_node_->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur_node_);
        node_stack_.push(node->children_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_[i] != node->children_[i + 1]) {
            node_stack_.push(node->children_[i]);
          }
        }
      }
    }

    AlexNode<T, P>* current() const { return cur_node_; }

    AlexNode<T, P>* next() {
      if (node_stack_.empty()) {
        cur_node_ = nullptr;
        return nullptr;
      }

      cur_node_ = node_stack_.top();
      node_stack_.pop();

      if (!cur_node_->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur_node_);
        node_stack_.push(node->children_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_[i] != node->children_[i + 1]) {
            node_stack_.push(node->children_[i]);
          }
        }
      }

      return cur_node_;
    }

    bool is_end() const { return cur_node_ == nullptr; }
  };
};
}
