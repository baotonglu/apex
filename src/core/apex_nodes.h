// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This file contains code for ALEX nodes. There are two types of nodes in ALEX:
 * - Model nodes (equivalent to internal/inner nodes of a B+ Tree)
 * - Data nodes, sometimes referred to as leaf nodes (equivalent to leaf nodes
 * of a B+ Tree)
 */

#pragma once

#include "apex_base.h"
#include "apex_log.h"
#include "../util/allocator.h"
#include <tuple>
#include <utility>
#include <algorithm>
#include <atomic>
#include <thread>
#include <map>
#include <tuple>
#include <vector>
#include <emmintrin.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
// Whether we store key and payload arrays separately in data nodes
// By default, we store them separately
//#define ALEX_DATA_NODE_SEP_ARRAYS 1

#if ALEX_DATA_NODE_SEP_ARRAYS
#define ALEX_DATA_NODE_KEY_AT(i) key_slots_[i]
#define ALEX_DATA_NODE_PAYLOAD_AT(i) payload_slots_[i]
#else
#define ALEX_DATA_NODE_KEY_AT(i) data_slots_[i].first
#define ALEX_DATA_NODE_PAYLOAD_AT(i) data_slots_[i].second
#endif

#define MY_PERSISTENCE 1
#define DRAM_BITMAP 1
#define HIDE_BITMAP 1 // hide the logic of bitmap operation
//#define NEW_LEAF 1

// Whether we use lzcnt and tzcnt when manipulating a bitmap (e.g., when finding
// the closest gap).
// If your hardware does not support lzcnt/tzcnt (e.g., your Intel CPU is
// pre-Haswell), set this to 0.
#define ALEX_USE_LZCNT 1

namespace alex {
// A parent class for both types of ALEX nodes
template <class T, class P>
class AlexNode {
 public:
  // Whether this node is a leaf (data) node
  bool is_leaf_ = false;

  // obsolete, to indicate whether this node has been deleted
  bool is_obsolete_ = false;
  // Power of 2 to which the pointer to this node is duplicated in its parent
  // model node
  // For example, if duplication_factor_ is 3, then there are 8 redundant
  // pointers to this node in its parent
  uint8_t local_depth_ = 0;

  // Node's level in the RMI. Root node is level 0
  uint8_t level_ = 0;

  // Both model nodes and data nodes nodes use models
  LinearModel<T> model_;

  // Could be either the expected or empirical cost, depending on how this field
  // is used
  double cost_ = 0.0;

  AlexNode() = default;
  explicit AlexNode(short level) : level_(level) {}
  AlexNode(short level, bool is_leaf) : is_leaf_(is_leaf), level_(level) {}
  virtual ~AlexNode() = default;

  // The size in bytes of all member variables in this class
  virtual long long node_size() const = 0;
};

template <class T, class P, class Alloc = my_alloc::allocator<std::pair<T, P>>>
class AlexModelNode : public AlexNode<T, P> {
 public:
  typedef AlexModelNode<T, P, Alloc> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<AlexNode<T, P>*>::other
      pointer_alloc_type;

  const Alloc& allocator_;

  // Lock, for synchronization of multiple threads
  uint32_t lock_ = 0;

  // Number of logical children. Must be a power of 2
  int num_children_ = 0;

  // Array of pointers to children
  // Hacking skills
  AlexNode<T, P>* children_[0];
  explicit AlexModelNode(const Alloc& alloc = Alloc())
      : AlexNode<T, P>(0, false), allocator_(alloc) {}

  explicit AlexModelNode(short level, const Alloc& alloc = Alloc())
      : AlexNode<T, P>(level, false), allocator_(alloc) {}

  ~AlexModelNode() {

  }

  // BT: FIXME, this should be crash consistent
  AlexModelNode(const self_type& other)
      : AlexNode<T, P>(other),
        allocator_(other.allocator_),
        num_children_(other.num_children_) {
    std::copy(other.children_, other.children_ + other.num_children_,
              children_);
  }

  static void New(PMEMoid *new_node, size_t num_children = 0){
    my_alloc::BasePMPool::Allocate(new_node, sizeof(AlexModelNode<T, P>) + num_children * sizeof(AlexNode<T, P>*));
    auto new_model_node = reinterpret_cast<AlexModelNode<T, P>*>(pmemobj_direct(*new_node));
    new_model_node->num_children_ = num_children;
  }

  uint64_t get_node_size() {
    //return (sizeof(AlexModelNode<T, P>) + num_children_ * sizeof(AlexNode<T, P>*));
    return num_children_ * sizeof(AlexNode<T, P>*);
  }

  // Given a key, traverses to the child node responsible for that key
  inline AlexNode<T, P>* get_child_node(const T& key) {
    int bucketID = this->model_.predict(key);
    bucketID = std::min<int>(std::max<int>(bucketID, 0), num_children_ - 1);
    return children_[bucketID];
  }

  // The RW Lock
  // Only the node expansion of the model node needs to get the write lock, otherwise
  // The read lock is enough
  inline bool get_write_lock(){
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    if ((v & lockSet) ||  this->is_obsolete_)
    {
      return false;
    }
    uint32_t old_value = v & lockMask;
    uint32_t new_value = old_value | lockSet;

    while (!CAS(&lock_, &old_value, new_value)) {
      if ((old_value & lockSet) ||  this->is_obsolete_)
      {
        return false;
      }
      old_value = old_value & lockMask;
      new_value = old_value | lockSet;
    }

    //wait until the readers all exit the critical section
    v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    while(v & lockMask){
      v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    }
    return true;
  }

  inline bool promote_from_read_to_write(){
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    if ((v & lockSet) ||  this->is_obsolete_)
    {
      return false;
    }

    uint32_t old_value = v & lockMask;
    uint32_t new_value = old_value | lockSet;
    while (!CAS(&lock_, &old_value, new_value)) {
      if ((old_value & lockSet) ||  this->is_obsolete_)
      {
        return false;
      }
      old_value = old_value & lockMask;
      new_value = old_value | lockSet;
    }

    //Decrease the #readers by 1, since it has been promoted to writer lock
    SUB(&lock_, 1);

    v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    while(v & lockMask){
      v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    }
    return true;
  }

  inline bool try_get_write_lock(){
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    uint32_t old_value = v & lockMask;
    uint32_t new_value = old_value | lockSet;

    if (!CAS(&lock_, &old_value, new_value)) {
      return false;
    }

    //wait until the readers all exit the critical section
    v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    while(v & lockMask){
      v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    }

    return true;
  }

  inline bool test_write_lock_set(){
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    return v & lockSet;
  }

  inline bool get_read_lock(){
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    if((v & lockSet) || this->is_obsolete_) return false;

    uint32_t old_value = v & lockMask;
    auto new_value = ((v & lockMask) + 1) & lockMask; 
    while (!CAS(&lock_, &old_value, new_value)){
      if((old_value & lockSet) || this->is_obsolete_){
        return false;
      }
      old_value = old_value & lockMask;
      new_value = ((old_value & lockMask) + 1) & lockMask; 
    }

    return true;
  }

  inline bool try_get_read_lock(){
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    if((v & lockSet) || this->is_obsolete_) return false;
    uint32_t old_value = v & lockMask;
    auto new_value = ((v & lockMask) + 1) & lockMask;
    return CAS(&lock_, &old_value, new_value);
  }

  inline void release_read_lock(){
    SUB(&lock_, 1);
  }

  inline void reset_rw_lock(){ 
    lock_ = 0;
    clwb(&lock_);
    sfence();
  }

  inline void release_write_lock(){
    __atomic_store_n(&lock_, 0, __ATOMIC_RELEASE);
  }

  void display_all_child(){ 
    printf("All children of node %p\n", this);
    for(int i = 0; i < num_children_; ++i){
      printf("child %d: %p; ", i, children_[i]);
    }
    std::cout << std::endl;
  }

  // Expand by a power of 2 by creating duplicates of all existing child
  // pointers.
  // Input is the base 2 log of the expansion factor, in order to guarantee
  // expanding by a power of 2.
  // Returns the expansion factor.
  int expand(int log2_expansion_factor) {
    assert(log2_expansion_factor >= 0);
    int expansion_factor = 1 << log2_expansion_factor;
    int num_new_children = num_children_ * expansion_factor;
    std::cout << "Error: expand should not be invoked here" << std::endl;
    exit(-1);
    return expansion_factor;
  }

  // Expansion based on old node, used in node split
  // So the expansion factor could only be the 2 power
  static void expand(PMEMoid *new_node, AlexModelNode<T,P> *old_node, int log2_expansion_factor){
    auto callback = [](PMEMobjpool *pool, void *ptr, void *arg){
      auto value_ptr = reinterpret_cast<std::tuple<int, AlexModelNode<T,P>*> *>(arg);
      auto node_ptr = reinterpret_cast<AlexModelNode<T, P>*>(ptr);
      auto log2_expansion_factor = std::get<0>(*value_ptr);
      auto old_node = std::get<1>(*value_ptr);

      //memset(node_ptr, 0, sizeof(AlexModelNode<T, P>));
      int expansion_factor = 1 << log2_expansion_factor;
      auto old_num_children = old_node->num_children_;
      auto old_children = old_node->children_;

      node_ptr->num_children_ = old_num_children * expansion_factor;
      node_ptr->lock_ = old_node->lock_;
      node_ptr->is_leaf_ = false;
      node_ptr->is_obsolete_ = false;
      node_ptr->local_depth_ = old_node->local_depth_;
      node_ptr->level_ = old_node->level_;
      node_ptr->cost_ = old_node->cost_;
      node_ptr->model_ = old_node->model_;
      // model_expand
      node_ptr->model_.expand(expansion_factor);

      for(int i = 0; i < old_num_children; ++i){
        for(int j = 0; j < expansion_factor; ++j){
          node_ptr->children_[i * expansion_factor + j] = old_children[i];
        }
      }

      pmemobj_persist(pool, node_ptr, sizeof(AlexModelNode<T,P>) + sizeof(AlexNode<T,P>*) * old_num_children * expansion_factor);
      return 0;
    };
    auto callback_args = std::make_tuple(log2_expansion_factor, old_node);
    my_alloc::BasePMPool::Allocate(new_node, CACHE_LINE_SIZE, sizeof(AlexModelNode<T, P>) + sizeof(AlexNode<T, P>*) * old_node->num_children_ * (1 << log2_expansion_factor), callback, reinterpret_cast<void*>(&callback_args));
  }

  pointer_alloc_type pointer_allocator() {
    return pointer_alloc_type(allocator_);
  }

  long long node_size() const override {
    long long size = sizeof(self_type);
    size += num_children_ * sizeof(AlexNode<T, P>*);  // pointers to children
    return size;
  }
};

/*
* Functions are organized into different sections:
* - Constructors and destructors
* - General helper functions
* - Iterator
* - Cost model
* - Bulk loading and model building (e.g., bulk_load, bulk_load_from_existing)
* - Lookups (e.g., find_key, find_lower, find_upper, lower_bound, upper_bound)
* - Inserts and resizes (e.g, insert)
* - Deletes (e.g., erase, erase_one)
* - Stats
* - Debugging
*/
template <class T, class P, class Compare = AlexCompare,
          class Alloc = my_alloc::allocator<std::pair<T, P>>,
          bool allow_duplicates = true>
class AlexDataNode : public AlexNode<T, P> {
 public:
  typedef std::pair<T, P> V;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<T>::other key_alloc_type;
  typedef typename Alloc::template rebind<P>::other payload_alloc_type;
  typedef typename Alloc::template rebind<V>::other value_alloc_type;
  typedef typename Alloc::template rebind<uint64_t>::other bitmap_alloc_type;

  const Compare& key_less_;
  // Forward declaration
  template <typename node_type = self_type, typename payload_return_type = P,
            typename value_return_type = V>
  class Iterator;
  typedef Iterator<> iterator_type;
  typedef Iterator<const self_type, const P, const V> const_iterator_type;

  template <typename node_type = self_type, typename payload_return_type = P,
            typename value_return_type = V>
  class OrderIterator;
  typedef OrderIterator<> o_iterator_type;
  typedef OrderIterator<const self_type, const P, const V> order_iterator_type;
  
  template <typename key_type = T, typename payload_type = P>
  class OverflowStash;
  typedef OverflowStash<T, P> overflow_stash_type;

  template <typename key_type = T, typename payload_type = P>
  class IndexOverflowStash;
  typedef IndexOverflowStash<T, P> index_overflow_stash_type;

  // 128 byte (2 cacheline size)
  class OverflowFinger{
    public:
    // Use header 16 bits to store the fingerprint of the key-value
    V* overflow_array_[OVERFLOW_FINGER_LENGTH];
    //uint16_t offset_[OVERFLOW_FINGER_LENGTH]; // If the first bit is set, indicate this kV is stored in overflow stash, otherwise, it stores the offset in stash array 
    OverflowFinger *next_; 

    inline uint32_t get_bitmap() const { 
      uint32_t bitmap = 0;
      for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
        if(overflow_array_[i] != nullptr){
          bitmap |= (1U << i); 
        }
      }
      return bitmap;    
    }
  };

  // 28 byte byte
  class MetaInfo{
 public:
    inline uint32_t get_bitmap() const {
      uint64_t bitmap = reinterpret_cast<uint64_t>(overflow_finger_) >> 48; 
      return static_cast<uint32_t>(bitmap);
    }

    // pos = 0...15
    inline void set_bitmap(int pos) {
      uint64_t bitmap = 1ULL << (pos + 48);
      overflow_finger_ = reinterpret_cast<OverflowFinger*>(reinterpret_cast<uint64_t>(overflow_finger_) | bitmap);
    }

    inline void unset_bitmap(int pos){
      uint64_t bitmap = ~(1ULL << (pos + 48));
      overflow_finger_ = reinterpret_cast<OverflowFinger*>(reinterpret_cast<uint64_t>(overflow_finger_) & bitmap);
    }

    // Need to shift 48 bites before udpating the bitmap
    inline void update_bitmap(uint64_t bitmap){
      overflow_finger_ = reinterpret_cast<OverflowFinger*>((reinterpret_cast<uint64_t>(overflow_finger_) & addrMask) | (bitmap << 48));
    }

    inline OverflowFinger* get_overflow_finger(){ 
      return reinterpret_cast<OverflowFinger*>(reinterpret_cast<uint64_t>(overflow_finger_) & addrMask);
    }

    inline void set_overflow_finger(OverflowFinger *new_finger){ 
      overflow_finger_ = reinterpret_cast<OverflowFinger*>((reinterpret_cast<uint64_t>(overflow_finger_) & headerMask) | reinterpret_cast<uint64_t>(new_finger)); 
    }

    OverflowFinger *overflow_finger_; // The bitmap is stored in the heading 16 bits of the pointer
    char fingerprint_[TABLE_FACTOR];
  };

  // To scale on multi-core, we partition these metadata for every SCALE_FACTOR records
  // 64 bytes
  class ScaleParameter{
    public: 
    int num_keys_ = 0;
    int num_lookups_ = 0;
    int num_inserts_ = 0;
    int lock_ = 0; // 16B
    double num_insert_cost_ = 0;
    double num_search_cost_ = 0; // 32B
    index_overflow_stash_type *index_overflow_stash_ = nullptr;
    overflow_stash_type *last_stash_ = nullptr;
    int overflow_stash_count_ = 0;
    int expansion_threshold_ = 0; // 48B
    int array_insert_ = 0; 
    int stash_insert_ = 0; //56B
  };

  MetaInfo* meta_info_array_ = nullptr;
  ScaleParameter* scale_parameters_ = nullptr;

  uint32_t local_version_ = 0; // Used to record whether this node needs to be recoverred
  uint32_t lock_ = 0;
  int data_capacity_ = 0;  // size of key/data_slots array
  int scale_factor_ = SCALE_FACTOR;
#if ALEX_DATA_NODE_SEP_ARRAYS
  T* key_slots_ = nullptr;  // holds keys
  P* payload_slots_ =
      nullptr;  // holds payloads, must be same size as key_slots
#else
  V* data_slots_ = nullptr;  // holds key-payload pairs
#endif
  int num_keys_ = 0;  // number of filled key/data slots (as opposed to gaps)
  int stash_capacity_ = 0;

// Variables for determining append-mostly behavior
  T max_key_ = std::numeric_limits<
      T>::lowest();  // max key in node, updates after inserts but not erases
  T min_key_ = std::numeric_limits<T>::max();  // min key in node, updates after
                                               // inserts but not erases
  int num_right_out_of_bounds_inserts_ =
      0;  // number of inserts that are larger than the max key
  int num_left_out_of_bounds_inserts_ =
      0;  // number of inserts that are smaller than the min key

  V *stash_slots_ = nullptr;
  int scale_parameters_size_ = 0;
  int bitmap_size_ = 0;
  uint64_t* bitmap_ = nullptr; 

  // Variables related to resizing (expansions and contractions)
  static constexpr double kMaxDensity_ = 0.9;  // density after contracting,
                                               // also determines the expansion
                                               // threshold
  static constexpr double kInitDensity_ =
      0.7;  // density of data nodes after bulk loading
  static constexpr double kMinDensity_ = 0.5;  // density after expanding, also
                                               // determines the contraction
                                               // threshold
  double expansion_threshold_ = 1;  // expand after m_num_keys is >= this number
  double contraction_threshold_ =
      0;  // contract after m_num_keys is < this number
  static constexpr int kDefaultMaxDataNodeBytes_ =
      1 << 18;  // by default, maximum data node size is 16MB
  int max_slots_ =
      kDefaultMaxDataNodeBytes_ /
      sizeof(V);  // cannot expand beyond this number of key/data slots

  // Used to determine the range of the data in this node
  // Note that the first node and last node may violate this limit becuase of out-of-bound insert
  double max_limit_ = 0;
  double min_limit_ = 0;

  // Counters used in cost models
#ifndef NEW_COST_MODEL      
  long long num_shifts_ = 0;                 // does not reset after resizing
  long long num_exp_search_iterations_ = 0;  // does not reset after resizing
#else
  // My new cost model
  double num_insert_cost_ = 0; // including insert in nornaml array, in overflow list, create overflow list
  double num_search_cost_ = 0; // mainly calculate the cache misses of difference search distance
#endif

  // important for recovery, set during bulk load or initialization
  T invalid_key_ = 0;

  int num_lookups_ = 0;  // does not reset after resizing
  int num_inserts_ = 0;  // does not reset after resizing

  int real_data_capacity_ = 0; // to enable linear probing, I will add probing_length elements at the end of the array
  int link_bitmap_size_ = 0;
  uint32_t link_lock_ = 0;
  overflow_stash_type *first_block_ = nullptr; 
  V* sorted_slots_ = nullptr; // This is used to build sorted array in the system

  // Node is considered append-mostly if the fraction of inserts that are out of
  // bounds is above this threshold
  // Append-mostly nodes will expand in a manner that anticipates further
  // appends
  static constexpr double kAppendMostlyThreshold = 0.9;

  // Purely for benchmark debugging purposes
#ifndef NEW_COST_MODEL  
  double expected_avg_exp_search_iterations_ = 0;
  double expected_avg_shifts_ = 0;
#else  
  double expected_avg_search_cost_ = 0;
  double expected_avg_insert_cost_ = 0;
#endif

  self_type* next_leaf_ = nullptr;
  self_type* prev_leaf_ = nullptr;

#if ALEX_DATA_NODE_SEP_ARRAYS
  PMEMoid P_key_slots_ = OID_NULL;
  PMEMoid P_payload_slots_ = OID_NULL;
#else  
  PMEMoid P_data_slots_ = OID_NULL; // used for data slots alloc and also for overflow stash alloc
#endif
  PMEMmutex recover_lock_;
  const Alloc& allocator_;

  /*** Constructors and destructors ***/

  explicit AlexDataNode(const Compare& comp = Compare(),
                        const Alloc& alloc = Alloc())
      : AlexNode<T, P>(0, true), key_less_(comp), allocator_(alloc) {
      }

  AlexDataNode(T invalid_key, const Compare& comp = Compare(),
                        const Alloc& alloc = Alloc())
      : AlexNode<T, P>(0, true), invalid_key_(invalid_key), key_less_(comp), allocator_(alloc) {
      }

  AlexDataNode(short level, int max_data_node_slots,
               const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : AlexNode<T, P>(level, true),
        key_less_(comp),
        allocator_(alloc),
        max_slots_(max_data_node_slots) {}

  ~AlexDataNode() {
    if(sorted_slots_ != nullptr){
        free(sorted_slots_);
        sorted_slots_ = nullptr;
    }
#if ALEX_DATA_NODE_SEP_ARRAYS
#ifdef LOG_SMO
    if (!OID_IS_NULL(P_key_slots_)) {
      pmemobj_free(&P_key_slots_);
    }

    if(!OID_IS_NULL(P_payload_slots_)){
      pmemobj_free(&P_payload_slots_);
    }
#else
    if (key_slots_ == nullptr) {
      return;
    }
    key_allocator().deallocate(key_slots_, real_data_capacity_);
    payload_allocator().deallocate(payload_slots_, real_data_capacity_);
#endif
#else
    if(!OID_IS_NULL(P_data_slots_)){
      pmemobj_free(&P_data_slots_);
    }
#endif 

    if(meta_info_array_ == nullptr) return;
    deallocate_all_overflow_stash(scale_parameters_, scale_parameters_size_);
    deallocate_all_overflow_finger(meta_info_array_, real_data_capacity_ / TABLE_FACTOR + 1);
    
    delete [] meta_info_array_;
    delete [] scale_parameters_;
    delete [] bitmap_;
    // delete the link bitmap 
  }

  void safe_destory(){
    if(sorted_slots_ != nullptr){
        free(sorted_slots_);
        sorted_slots_ = nullptr;
    }

    if(data_slots_ != nullptr){
      auto ptr = pmemobj_oid(data_slots_);
      pmemobj_free(&ptr);
    }

    if(meta_info_array_ == nullptr) return;
    deallocate_all_overflow_stash(scale_parameters_, scale_parameters_size_);
    deallocate_all_overflow_finger(meta_info_array_, real_data_capacity_ / TABLE_FACTOR + 1);

    delete [] meta_info_array_;
    delete [] scale_parameters_;
    delete [] bitmap_;
  }

  // Upon recovery, only PM storage exists
  void recover_reclaim(){
    // Only deallocate the PM storage
    void *ptr = pmemobj_direct(P_data_slots_);
    if((ptr != reinterpret_cast<void*>(data_slots_)) && (ptr != reinterpret_cast<void*>(first_block_))){
      pmemobj_free(&P_data_slots_);
    }

    if(data_slots_ != nullptr){
      my_alloc::BasePMPool::Free(data_slots_);
      data_slots_ = nullptr;
    }

    overflow_stash_type *link_stash = first_block_;
    while(link_stash){
      overflow_stash_type *next_stash = link_stash;
      link_stash = link_stash->link_;
      while(!OID_IS_NULL(next_stash->next_)){
        overflow_stash_type *cur_stash = next_stash;
        next_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(cur_stash->next_));
        my_alloc::BasePMPool::Free(cur_stash);
      }
      my_alloc::BasePMPool::Free(next_stash); 
    }
  }

  // FIXME(BT): this is not crash consistent
  AlexDataNode(const self_type& other)
      : AlexNode<T, P>(other),
        key_less_(other.key_less_),
        allocator_(other.allocator_),
        next_leaf_(other.next_leaf_),
        prev_leaf_(other.prev_leaf_),
        data_capacity_(other.data_capacity_),
        real_data_capacity_(other.real_data_capacity_),
        num_keys_(other.num_keys_),
        bitmap_size_(other.bitmap_size_),
        expansion_threshold_(other.expansion_threshold_),
        contraction_threshold_(other.contraction_threshold_),
        max_slots_(other.max_slots_),
#ifndef NEW_COST_MODEL        
        num_shifts_(other.num_shifts_),
        num_exp_search_iterations_(other.num_exp_search_iterations_),
#else
        num_search_cost_(other.num_search_cost_),
        num_insert_cost_(other.num_insert_cost_),
#endif
        num_lookups_(other.num_lookups_),
        num_inserts_(other.num_inserts_),
        max_key_(other.max_key_),
        min_key_(other.min_key_),
        num_right_out_of_bounds_inserts_(
            other.num_right_out_of_bounds_inserts_),
        num_left_out_of_bounds_inserts_(other.num_left_out_of_bounds_inserts_),
#ifdef NEW_COST_MODEL
        expected_avg_search_cost_(
            other.expected_avg_search_cost_),
        expected_avg_insert_cost_(other.expected_avg_insert_cost_) 
#else        
        expected_avg_exp_search_iterations_(
            other.expected_avg_exp_search_iterations_),
        expected_avg_shifts_(other.expected_avg_shifts_) 
#endif
{
#ifdef LOG_SMO
#if ALEX_DATA_NODE_SEP_ARRAYS
    my_alloc::BasePMPool::Allocate(&P_key_slots_, sizeof(T)*real_data_capacity_);
    key_slots_ = reinterpret_cast<T*>(pmemobj_direct(P_key_slots_));
    std::copy(other.key_slots_, other.key_slots_ + other.real_data_capacity_,
              key_slots_);
    my_alloc::BasePMPool::Allocate(&P_payload_slots_, sizeof(P)*real_data_capacity_);
    payload_slots_ = reinterpret_cast<P*>(pmemobj_direct(P_payload_slots_));
    std::copy(other.payload_slots_, other.payload_slots_ + other.real_data_capacity_,
              payload_slots_);
#else
    my_alloc::BasePMPool::Allocate(&P_data_slots_, sizeof(V)*real_data_capacity_);
    data_slots_ = reinterpret_cast<V*>(pmemobj_direct(P_data_slots_));
    std::copy(other.data_slots_, other.data_slots_ + other.real_data_capacity_,
              data_slots_);
#endif
#else
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ = new (key_allocator().allocate(other.real_data_capacity_))
        T[other.real_data_capacity_];
    std::copy(other.key_slots_, other.key_slots_ + other.real_data_capacity_,
              key_slots_);
    payload_slots_ = new (payload_allocator().allocate(other.real_data_capacity_))
        P[other.real_data_capacity_];
    std::copy(other.payload_slots_, other.payload_slots_ + other.real_data_capacity_,
              payload_slots_);
#else
    data_slots_ = new (value_allocator().allocate(other.real_data_capacity_))
        V[other.real_data_capacity_];
    std::copy(other.data_slots_, other.data_slots_ + other.real_data_capacity_,
              data_slots_);
#endif
#endif

    //meta_info_array_ = new MetaInfo[other.real_data_capacity_ / TABLE_FACTOR + 1];
    auto metainfo_num = other.real_data_capacity_ / TABLE_FACTOR + 1;
    align_zalloc((void**)&meta_info_array_, sizeof(MetaInfo) * metainfo_num);
    std::copy(other.meta_info_array_, other.meta_info_array_ + metainfo_num, meta_info_array_);
   
    sorted_slots_ = nullptr;
    lock_ = 0;
  }
  /*** Memory management***/

  static void Free(self_type * node){
    auto callback = [](void *callback_context, void *ptr) {
      auto object = reinterpret_cast<self_type*>(ptr);
      object->safe_destory();
      auto oid_ptr = pmemobj_oid(ptr);
      pmemobj_free(&oid_ptr);
    };
    my_alloc::BasePMPool::SafeFree(reinterpret_cast<void*>(node), callback);
  }

  /*** concurrency management **/
  inline void get_lock() {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
      while (true) {
        old_value = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
        if (!(old_value & lockSet)) {
          old_value &= lockMask;
          break;
        }
      }
      new_value = old_value | lockSet;
    } while (!CAS(&lock_, &old_value, new_value));
  }

  inline bool try_get_lock() {
    uint32_t v = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    if (v & lockSet) {
      return false;
    }
    auto old_value = v & lockMask;
    auto new_value = v | lockSet;
    return CAS(&lock_, &old_value, new_value);
  }

  inline void release_lock() {
    uint32_t v = lock_;
    __atomic_store_n(&lock_, v + 1 - lockSet, __ATOMIC_RELEASE);
  }

  inline void get_link_lock(){
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
      while (true) {
        old_value = __atomic_load_n(&link_lock_, __ATOMIC_ACQUIRE);
        if (!(old_value & lockSet)) {
          old_value &= lockMask;
          break;
        }
      }
      new_value = old_value | lockSet;
    } while (!CAS(&link_lock_, &old_value, new_value));
  }

  inline bool try_get_link_lock(){
    uint32_t new_value = 0;
    uint32_t old_value = __atomic_load_n(&link_lock_, __ATOMIC_ACQUIRE);
    if(old_value & lockSet) return false;
    old_value &= lockMask;
    new_value = old_value | lockSet;
    return CAS(&link_lock_, &old_value, new_value);
  }

  inline void release_link_lock() {
    uint32_t v = link_lock_;
    __atomic_store_n(&link_lock_, v & lockMask, __ATOMIC_RELEASE);
  }

  /*if the lock is set, return true*/
  inline bool test_lock_set(uint32_t &version) const {
    version = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    return (version & lockSet) != 0;
  }

  // test whether the version has change, if change, return true
  inline bool test_lock_version_change(uint32_t old_version) const {
    auto value = __atomic_load_n(&lock_, __ATOMIC_ACQUIRE);
    return (old_version != value);
  }

  /*** Fine-grained concurrency control ***/
  inline bool test_lock_set(uint32_t &version, int lock_pos) const {
    version = __atomic_load_n(&scale_parameters_[lock_pos].lock_, __ATOMIC_ACQUIRE);
    return (version & lockSet) != 0;
  }

  inline bool test_lock_version_change(uint32_t old_version, int lock_pos) const {
    auto value = __atomic_load_n(&scale_parameters_[lock_pos].lock_, __ATOMIC_ACQUIRE);
    return (old_version != value);
  }

  inline void get_lock(int lock_pos) {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
      while (true) {
        old_value = __atomic_load_n(&scale_parameters_[lock_pos].lock_, __ATOMIC_ACQUIRE);
        if (!(old_value & lockSet)) {
          old_value &= lockMask;
          break;
        }
      }
      new_value = old_value | lockSet;
    } 
    while (!CAS(&scale_parameters_[lock_pos].lock_, &old_value, new_value));
  }

  inline bool try_get_lock(int lock_pos) {
    uint32_t v = __atomic_load_n(&scale_parameters_[lock_pos].lock_, __ATOMIC_ACQUIRE);
    if (v & lockSet) {
      return false;
    }
    auto old_value = v & lockMask;
    auto new_value = v | lockSet;
    return CAS(&scale_parameters_[lock_pos].lock_, &old_value, new_value);
  }

  inline void release_lock(int lock_pos) {
    uint32_t v = scale_parameters_[lock_pos].lock_;
    __atomic_store_n(&scale_parameters_[lock_pos].lock_, v + 1 - lockSet, __ATOMIC_RELEASE);
  }

  /*** Allocators ***/

  key_alloc_type key_allocator() { return key_alloc_type(allocator_); }

  payload_alloc_type payload_allocator() {
    return payload_alloc_type(allocator_);
  }

  value_alloc_type value_allocator() { return value_alloc_type(allocator_); }

  /*** General helper functions ***/

  inline T& get_key(int pos) const { return ALEX_DATA_NODE_KEY_AT(pos); }

  inline P& get_payload(int pos) const {
    return ALEX_DATA_NODE_PAYLOAD_AT(pos);
  }

  // Check whether the position corresponds to a key (as opposed to a gap)
  inline bool check_exists(int pos) const {    
    assert(pos >= 0 && pos < real_data_capacity_);
    int array_pos = pos / TABLE_FACTOR;
    int internal_pos = pos & table_factor_hide;
    return static_cast<bool>(meta_info_array_[array_pos].get_bitmap() & (1U << internal_pos));
  }

  // Value of first (i.e., min) key
  T first_key() const {
    order_iterator_type it(this);
    if(!it.is_end()){
      return it.key();
    }
    return std::numeric_limits<T>::max();
  }

  // Value of last (i.e., max) key
  T last_key() const {
    order_iterator_type it(this);
    if(!it.is_end()){
      return it.last_key();
    }
    return std::numeric_limits<T>::lowest();
  }

  // Position in key/data_slots of first (i.e., min) key
  int first_pos() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Position in key/data_slots of last (i.e., max) key
  int last_pos() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Number of keys between positions left and right (exclusive) in
  // key/data_slots
  int num_keys_in_range(int left, int right, bool use_sorted_node = false) const {
    if(use_sorted_node) return (right - left);
    printf("Should not not incur this!!\n");
    return -1;
  }

  // True if a < b
  template <class K>
  forceinline bool key_less(const T& a, const K& b) const {
    return key_less_(a, b);
  }

  // True if a <= b
  template <class K>
  forceinline bool key_lessequal(const T& a, const K& b) const {
    return !key_less_(b, a);
  }

  // True if a > b
  template <class K>
  forceinline bool key_greater(const T& a, const K& b) const {
    return key_less_(b, a);
  }

  // True if a >= b
  template <class K>
  forceinline bool key_greaterequal(const T& a, const K& b) const {
    return !key_less_(a, b);
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const T& a, const K& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Iterator ***/


// New design of iterator
// The iterator first iterator the key-value items in the main data array; then iterates the items overflow data blocks
// This iterator has more cache misses because it iterates the stash array for two times 
template <typename node_type, typename payload_return_type,
            typename value_return_type>
  class Iterator {
public:
    node_type* node_;
    int cur_idx_ = 0;  // current position in key/data_slots, -1 if at end; also used as the position in metadata_array, -1 if at end
    int cur_idx_in_block_ = 0;
    int cur_bitmap_idx_ = 0;  // current index position in bitmap
    int bitmap_size_ = 0;
    uint32_t cur_bitmap_data_ =
        0;  // caches the relevant data in the current bitmap position
    bool scanning_linked_list = false; // the kv records in linked list are stored in a seperate array in DRAM
    OverflowFinger *cur_overflow_finger =nullptr;

    explicit Iterator(node_type* node) : node_(node) {
      scanning_linked_list = false;
      initialize();
    }

    Iterator(node_type* node, int idx) : node_(node), cur_idx_(idx) {
      scanning_linked_list = false;
      initialize();
    }

    void initialize() {
      // new iterator
      cur_overflow_finger = nullptr;
      cur_bitmap_idx_ = cur_idx_ / TABLE_FACTOR;
      cur_bitmap_data_ = node_->meta_info_array_[cur_bitmap_idx_].get_bitmap();
      int start_internal_pos = cur_idx_ & table_factor_hide; //internal shift
      cur_bitmap_data_ &= ~((1U << start_internal_pos) - 1);
      bitmap_size_ = node_->real_data_capacity_ / TABLE_FACTOR + 1;

      (*this)++;
    }

    void operator++(int) {
      if(!scanning_linked_list){
        while (cur_bitmap_data_ == 0) {
          cur_bitmap_idx_++;
          if (cur_bitmap_idx_ >= bitmap_size_) {
            scanning_linked_list = true;
            break;
          }
          cur_bitmap_data_ = node_->meta_info_array_[cur_bitmap_idx_].get_bitmap();
        }

        if(scanning_linked_list){
          locate_in_a_linked_list();
          return;
        }
              
        uint32_t bit = extract_rightmost_one(cur_bitmap_data_);
        cur_idx_ = cur_bitmap_idx_ * TABLE_FACTOR + count_ones(bit - 1);
        cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
      }else{
        locate_in_a_linked_list();
      }
    }

    OverflowFinger *next_overflow_finger(int cur_pos){
      for(int i = cur_pos; i < bitmap_size_; ++i){
        if(node_->meta_info_array_[i].get_overflow_finger() != nullptr){
            cur_idx_ = i;
            return node_->meta_info_array_[i].get_overflow_finger();
        }
      }
      cur_idx_ = -1;
      return nullptr;
    }

    void locate_in_a_linked_list(){ 
      if(cur_overflow_finger == nullptr){
        //First time, locate first block
        cur_overflow_finger = next_overflow_finger(0);
        // no overflow data
        if(cur_overflow_finger == nullptr) return;
        cur_idx_in_block_ = 0;
        cur_bitmap_data_ = cur_overflow_finger->get_bitmap();
      }

      while(cur_bitmap_data_ == 0){
        while((cur_overflow_finger->next_ != nullptr) && (cur_bitmap_data_ == 0)){
          cur_overflow_finger = cur_overflow_finger->next_;
          cur_bitmap_data_ = cur_overflow_finger->get_bitmap();
        }

        if(cur_bitmap_data_ == 0){
          cur_overflow_finger = next_overflow_finger(cur_idx_ + 1);
          if(cur_overflow_finger == nullptr) return;
          cur_bitmap_data_ = cur_overflow_finger->get_bitmap();
        }
      }
      
      uint32_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_in_block_ = count_ones(bit-1);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }

    value_return_type& operator*() const {
      if(!scanning_linked_list){
        return node_->data_slots_[cur_idx_];
      }else{
        return *(reinterpret_cast<value_return_type*>(addrMask & reinterpret_cast<uint64_t>(cur_overflow_finger->overflow_array_[cur_idx_in_block_])));
      }
    }

    const T& key() const {
      if(!scanning_linked_list){
        return node_->data_slots_[cur_idx_].first;
      }else{
        return (reinterpret_cast<value_return_type*>(addrMask & reinterpret_cast<uint64_t>(cur_overflow_finger->overflow_array_[cur_idx_in_block_])))->first;
      }
    }

    payload_return_type& payload() const {
      if(!scanning_linked_list){
        return node_->data_slots_[cur_idx_].second;
      }else{
        return (reinterpret_cast<value_return_type*>(addrMask & reinterpret_cast<uint64_t>(cur_overflow_finger->overflow_array_[cur_idx_in_block_])))->second;
      }
    }

    bool is_end() const { return cur_idx_ == -1; }

    bool operator==(const Iterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_;
    }

    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };
  };  

  // 64B, one cache line
  template <typename key_type, typename payload_type>
  class IndexOverflowStash{
  public:
    typedef std::pair<key_type, payload_type> V;
    uint16_t bitmap_[8]; //first bit indicated wheteher the block is allocated or not
    OverflowStash<key_type, payload_type>* overflow_stash_array_[5];
    IndexOverflowStash<key_type, payload_type> *next_;
  };

  inline V* index_overflow_stash_insert(index_overflow_stash_type *node, const T& key, const P& payload, T default_key, PMEMoid *tmp, int* ret_offset, int parameter_pos) { 
    uint16_t bitmap;
    V* ret = nullptr;
    *ret_offset = 0;
    for(int i = 0; i < 5; ++i){
      bitmap = node->bitmap_[i] | bitmapMask;
      if(bitmap & bitmapSet){
        if(bitmap != UINT16_MAX){
          // Find a free slot in existing overflow stash
          uint32_t bit = extract_rightmost_zero(static_cast<uint32_t>(bitmap));
          int offset = count_ones(bit - 1);
          overflow_stash_type *overflow_stash = node->overflow_stash_array_[i];
          ret = overflow_stash->overflow_slots_ + offset;
          ret->second = payload;
          ret->first = key;
          node->bitmap_[i] |= static_cast<uint16_t>(bit);
          *ret_offset = offset;
          break;
        }
      }else{
        // Create a new OverflowStash, need to ensure crash consistency
        overflow_stash_type::New(tmp, key, payload, default_key, parameter_pos);
        node->overflow_stash_array_[i] = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(*tmp));
        node->bitmap_[i] = ((uint16_t)1 << 15) | (uint16_t)1;
        ret = (node->overflow_stash_array_[i])->overflow_slots_;
        break;
      }
    }

    return ret;
  }

  // ret_offset could be used in multiple ways
  inline V* index_overflow_stash_insert_kv(index_overflow_stash_type *node, const T& key, const P& payload, int* ret_offset) { 
    uint16_t bitmap;
    V* ret = nullptr;
    *ret_offset = -1;
    for(int i = 0; i < 5; ++i){
      bitmap = node->bitmap_[i] | bitmapMask;
      if(bitmap & bitmapSet){
        if(bitmap != UINT16_MAX){
          // Find a free slot in existing overflow stash
          uint32_t bit = extract_rightmost_zero(static_cast<uint32_t>(bitmap));
          int offset = count_ones(bit - 1);
          overflow_stash_type *overflow_stash = node->overflow_stash_array_[i];
          ret = overflow_stash->overflow_slots_ + offset;
          ret->second = payload;
          ret->first = key;
          node->bitmap_[i] |= static_cast<uint16_t>(bit);
          *ret_offset = offset;
          break;
        }
      }else{
         *ret_offset = i; // The slot which could used to add another overflow_stash
        break;
      }
    }

    return ret;
  }

  inline V* index_overflow_stash_insert_alloc(index_overflow_stash_type *node, const T& key, const P& payload, T default_key, PMEMoid *tmp, int parameter_pos, int idx) { 
    overflow_stash_type::New(tmp, key, payload, default_key, parameter_pos);
    node->overflow_stash_array_[idx] = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(*tmp));
    node->bitmap_[idx] = ((uint16_t)1 << 15) | (uint16_t)1;
    return (node->overflow_stash_array_[idx])->overflow_slots_;
  }

  template <typename key_type, typename payload_type>
  class OverflowStash{
  public:
    typedef std::pair<key_type, payload_type> V;
    static void New(PMEMoid *tmp, key_type key, payload_type payload, key_type default_key, int parameter_pos) {
      auto callback = [](PMEMobjpool *pool, void *ptr, void *arg) {
        auto value_ptr = reinterpret_cast<std::tuple<key_type, payload_type, key_type, int> *>(arg);
        auto stash_ptr = reinterpret_cast<OverflowStash<key_type, payload_type>* >(ptr);
        key_type first_key = std::get<0>(*value_ptr);
        payload_type first_payload = std::get<1>(*value_ptr);
        key_type default_key = std::get<2>(*value_ptr);
        stash_ptr->parameter_pos_ = std::get<3>(*value_ptr);
        stash_ptr->overflow_slots_[0] = std::make_pair(first_key, first_payload);
        for(int i = 1; i < 11; ++i){
          stash_ptr->overflow_slots_[i].first = default_key;
        }
        pmemobj_persist(pool, stash_ptr, sizeof(OverflowStash<key_type, payload_type>));
        return 0;
      };
      std::tuple<key_type, payload_type, key_type, int> callback_para = {key, payload, default_key, parameter_pos};
      my_alloc::BasePMPool::Allocate(tmp, kCacheLineSize, sizeof(OverflowStash<key_type, payload_type>), callback,
                          reinterpret_cast<void *>(&callback_para));
    }

    inline uint16_t get_bitmap(key_type invalid_key){
      uint16_t bitmap = 0;
      for(int i = 0; i < 11; ++i){
        if(overflow_slots_[i].first != invalid_key){
          bitmap |= (static_cast<uint16_t>(1) << i);
        }
      }
      return bitmap;
    }

    V overflow_slots_[11];
    PMEMoid next_;
    OverflowStash<key_type, payload_type> *link_; // Link for consistency in node
    int parameter_pos_; /* which parameter pos this system belongs to*/
    char dummy[4]; // change this to parameter pos
  };

  // New get sorted array that only uses the bitmap in meta_array
    V* get_sorted_array() const{
    // Allocate a new array to store the sorted array
    void* tmp = malloc(num_keys_ * sizeof(V));
    V* sorted_slots = reinterpret_cast<V*>(tmp);
    if(tmp == NULL){
        std::cout << "Allocation Error for sorted slots" << std::endl;
        exit(-1);
    }
  
    V cur_key_payload;
    int i, j, arr_idx = -1; // arr_idx trakcing the index of sorted_slots
    int meta_capacity = real_data_capacity_ / TABLE_FACTOR + 1;
    uint32_t cur_bitmap_data;
    int cur_idx_;
    OverflowFinger *cur_overflow_finger;

    for(i = 0; i < meta_capacity; ++i){
      cur_bitmap_data = meta_info_array_[i].get_bitmap();
      // First part => scan the corresdnpoding data array
      while(cur_bitmap_data){
        uint32_t bit = extract_rightmost_one(cur_bitmap_data);
        cur_idx_ = i * TABLE_FACTOR + count_ones(bit - 1);
        cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
        cur_key_payload = data_slots_[cur_idx_];

        j = arr_idx;
        while ((j >= 0) && (key_greater(sorted_slots[j].first, cur_key_payload.first))){
          sorted_slots[j + 1] = sorted_slots[j];
          --j;
        }
        sorted_slots[j + 1] = cur_key_payload;
        arr_idx++;
      }

      // test whether it has overflow
      cur_overflow_finger = meta_info_array_[i].get_overflow_finger();
      while(cur_overflow_finger){
        cur_bitmap_data = cur_overflow_finger->get_bitmap();
        while(cur_bitmap_data){
          uint32_t bit = extract_rightmost_one(cur_bitmap_data);
          cur_idx_ = count_ones(bit - 1);
          cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
          cur_key_payload = *(reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(cur_overflow_finger->overflow_array_[cur_idx_])));

          j = arr_idx;
          while ((j >= 0) && (key_greater(sorted_slots[j].first, cur_key_payload.first))){
            sorted_slots[j + 1] = sorted_slots[j];
            --j;
          }

          sorted_slots[j + 1] = cur_key_payload;
          arr_idx++;
        }
        cur_overflow_finger = cur_overflow_finger->next_;
      }      
    }
    
    if((arr_idx + 1) != num_keys_){
      // Just for debugging
      std::cout << "arr_idx = " << arr_idx << "; num_keys = " << num_keys_ << std::endl;
      std::cout << "data capacity = " << data_capacity_ << std::endl;
      std::cout << "Count errror!!!" << std::endl;
      exit(-1);
    }

    return sorted_slots;
  }

  int total_keys_in_node(){ 
    int count = 0;
    for (int i = 0; i < scale_parameters_size_; i++)
    {
      count += scale_parameters_[i].num_keys_;
    }
    return count;
  }

  int stash_keys_in_node(){ 
    int count = 0;
    for (int i = 0; i < scale_parameters_size_; i++)
    {
      count += scale_parameters_[i].stash_insert_;
    }
    return count;
  }

  // Create a temporary sorted array, that are used for structure modification opeartion
  void build_sorted_slots(){ 
    num_keys_ = total_keys_in_node();
    if(num_keys_ == 0) return;
    if(sorted_slots_ != nullptr){
      free(sorted_slots_);
      sorted_slots_ = nullptr;
    }

    sorted_slots_ = get_sorted_array();
  }

// Order iterator that iterates from the smallest key to the largest key
  template <typename node_type, typename payload_return_type,
            typename value_return_type>
  class OrderIterator {
   public:
    node_type* node_;
    int cur_idx_ = 0;  // current position in key/data_slots, -1 if at end
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ =
        0;  // caches the relevant data in the current bitmap position
    V* sorted_array_ = nullptr; // the value is attached with the key, so I need to make a sort function
    int num_keys_ = 0;
    bool need_deallocate = false;
    
    OrderIterator(node_type* node, int set_arr_idx = 0, bool use_sorted_node = false) : node_(node) {
      num_keys_ = node->num_keys_;
      if(use_sorted_node){
        sorted_array_ = node->sorted_slots_;
      }else{
        sorted_array_ = node_->get_sorted_array();
        need_deallocate = true;
      }
      cur_idx_ = set_arr_idx;
    }

    ~OrderIterator(){
      if(need_deallocate && (sorted_array_ != nullptr)){
          delete sorted_array_;
      }
    }

    // Need to sort the array
    void initialize() {
    }

    void operator++(int) {
      cur_idx_++;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    V operator*() const {
      return sorted_array_[cur_idx_];
    }
#else
    value_return_type& operator*() const {
      return sorted_array_[cur_idx_];
    }
#endif

    const T& key() const {
#if ALEX_DATA_NODE_SEP_ARRAYS 
      return sorted_array_[cur_idx_].first;
#else
      return sorted_array_[cur_idx_].first;
#endif
    }

    const T& last_key() const { 
      return sorted_array_[num_keys_ - 1].first;
    }

    payload_return_type& payload() const {
#if ALEX_DATA_NODE_SEP_ARRAYS
      return sorted_array_[cur_idx_].second;
#else
      return sorted_array_[cur_idx_].second;
#endif
    }

    bool is_end() const { return cur_idx_ >= num_keys_; }

    bool operator==(const OrderIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_;
    }

    bool operator!=(const OrderIterator& rhs) const { return !(*this == rhs); };
  };

  // iterator_type begin() { return iterator_type(this, 0); }
  o_iterator_type begin() { return o_iterator_type(this, 0);}

  /*** Cost model ***/

  // Empirical average number of shifts per insert
#ifndef NEW_COST_MODEL  
  double shifts_per_insert() const {
    if (num_inserts_ == 0) {
      return 0;
    }
    return num_shifts_ / static_cast<double>(num_inserts_);
  }

    // Empirical average number of exponential search iterations per operation
  // (either lookup or insert)
  double exp_search_iterations_per_operation() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    return num_exp_search_iterations_ /
           static_cast<double>(num_inserts_ + num_lookups_);
  }

    double empirical_cost() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    double frac_inserts =
        static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
    return kExpSearchIterationsWeight * exp_search_iterations_per_operation() +
           kShiftsWeight * shifts_per_insert() * frac_inserts;
  }

#else  

  double insert_cost_per_insert() const {
    int total_inserts = num_inserts_;
    double total_insert_cost = num_insert_cost_;
    for(int i = 0; i < scale_parameters_size_; ++i){
      total_inserts += scale_parameters_[i].num_inserts_;
      total_insert_cost += scale_parameters_[i].num_insert_cost_;
    }

    if(total_inserts == 0){
      return 0;
    }
    return total_insert_cost / static_cast<double>(total_inserts);    
    //if(num_inserts_ == 0){
    //  return 0;
    //}
    //return num_insert_cost_ / static_cast<double>(num_inserts_);
  }

  double search_cost_per_operation() const {
    int total_inserts = num_inserts_;
    int total_lookups = num_lookups_;
    double total_insert_cost = num_insert_cost_;
    double total_search_cost = num_search_cost_;
    for(int i = 0; i < scale_parameters_size_; ++i){
      total_inserts += scale_parameters_[i].num_inserts_;
      total_lookups += scale_parameters_[i].num_lookups_;
      total_insert_cost += scale_parameters_[i].num_insert_cost_;
      total_search_cost += scale_parameters_[i].num_search_cost_;
    }

    if(total_inserts + total_lookups == 0){
      return 0;
    }

    return total_search_cost / static_cast<double>(total_inserts + total_lookups);
    /*
    if(num_inserts_ + num_lookups_ == 0){
      return 0;
    }

    return num_search_cost_ / static_cast<double>(num_inserts_ + num_lookups_);
    */
  }

  void update_all_cost(int *array_insert, int* stash_insert) {
    int total_inserts = 0;
    int total_lookups = 0;
    int total_array_insert = 0;
    int total_stash_insert = 0;
    int total_num_keys = 0;
    double total_insert_cost = 0;
    double total_search_cost = 0;
    for(int i = 0; i < scale_parameters_size_; ++i){
      total_inserts += scale_parameters_[i].num_inserts_;
      total_array_insert += scale_parameters_[i].array_insert_;
      total_stash_insert += scale_parameters_[i].stash_insert_;
      total_lookups += scale_parameters_[i].num_lookups_;
      total_insert_cost += scale_parameters_[i].num_insert_cost_;
      total_search_cost += scale_parameters_[i].num_search_cost_;
      total_num_keys += scale_parameters_[i].num_keys_;
    }

    num_inserts_ += total_inserts;
    num_lookups_ += total_lookups;
    num_search_cost_ += total_search_cost;
    num_insert_cost_ += total_insert_cost;
    num_keys_ = total_num_keys;
    *array_insert = total_array_insert;
    *stash_insert = total_stash_insert;
  }

  double new_empirical_cost() const {
    int total_inserts = num_inserts_;
    int total_lookups = num_lookups_;
    double total_insert_cost = num_insert_cost_;
    double total_search_cost = num_search_cost_;
    for(int i = 0; i < scale_parameters_size_; ++i){
      total_inserts += scale_parameters_[i].num_inserts_;
      total_lookups += scale_parameters_[i].num_lookups_;
      total_insert_cost += scale_parameters_[i].num_insert_cost_;
      total_search_cost += scale_parameters_[i].num_search_cost_;
    }

    if(total_inserts + total_lookups == 0){
      return 0;
    }

    double frac_inserts =
        static_cast<double>(total_inserts) / (total_inserts + total_lookups);
    double avg_search_cost = total_search_cost / static_cast<double>(total_inserts + total_lookups);
    double avg_insert_cost = total_insert_cost / static_cast<double>(total_inserts); 
    return kSearchCostWeight * avg_search_cost +
           kInsertCostWeight * avg_insert_cost * frac_inserts;
/*
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    double frac_inserts =
        static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
    return kSearchCostWeight * search_cost_per_operation() +
           kInsertCostWeight * insert_cost_per_insert() * frac_inserts;
*/
  }

#endif  //NEW_COST_MODEL

  // Empirical fraction of operations (either lookup or insert) that are inserts
  double frac_inserts() const {
    int total_inserts = num_inserts_;
    int total_lookups = num_lookups_;
    for(int i = 0; i < scale_parameters_size_; ++i){
      total_inserts += scale_parameters_[i].num_inserts_;
      total_lookups += scale_parameters_[i].num_lookups_;
    }

    if(total_inserts + total_lookups == 0){
      return 0;
    }

    return  static_cast<double>(total_inserts) / (total_inserts + total_lookups);
/*
    int num_ops = num_inserts_ + num_lookups_;
    if (num_ops == 0) {
      return 0;  // if no operations, assume no inserts
    }
    return static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
*/
  }

  void reset_stats() {
#ifndef NEW_COST_MODEL    
    num_shifts_ = 0;
    num_exp_search_iterations_ = 0;
#else
    num_insert_cost_ = 0;
    num_search_cost_ = 0;
#endif
    num_lookups_ = 0;
    num_inserts_ = 0;
  }

  void print_all_data_in_order(){
    order_iterator_type it(this);
    for(; !it.is_end(); it++){
      printf("%.8lf ", it.key());
    }
    std::cout << std::endl;
  }

  // Computes the expected cost of the current node
  double compute_expected_cost(double frac_inserts = 0) {
    if (num_keys_ == 0) {
      return 0;
    }

    ExpectedInsertAndSearchCostAccumulator search_insert_accumulator(sizeof(V), 13);
    int meta_size = real_data_capacity_ / TABLE_FACTOR + 1;
    uint32_t cur_bitmap_data;
    int cur_idx_;
    for(int i = 0; i < meta_size; ++i){
      cur_bitmap_data = meta_info_array_[i].get_bitmap();
      // First part => scan the corresdnpoding data array
      while(cur_bitmap_data){
        uint32_t bit = extract_rightmost_one(cur_bitmap_data);
        cur_idx_ = i * TABLE_FACTOR + count_ones(bit - 1);
        cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
        int predicted_position = std::max(
        0, std::min(data_capacity_ - 1, this->model_.predict(data_slots_[cur_idx_].first)));
        search_insert_accumulator.accumulate(cur_idx_, predicted_position);
      }

      OverflowFinger *overflow_finger = meta_info_array_[i].get_overflow_finger();
      int overflow_num = 0;
      while(overflow_finger != nullptr){
        V** overflow_array = overflow_finger->overflow_array_;
        for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
          if(overflow_array[i] != nullptr){
            ++overflow_num;
            search_insert_accumulator.overflow_accumulate(overflow_num);
          }
        }
        overflow_finger = overflow_finger->next_;
      }
    }

    expected_avg_search_cost_ = search_insert_accumulator.get_expected_search_cost();
    expected_avg_insert_cost_ = search_insert_accumulator.get_expected_insert_cost();
    double cost = kSearchCostWeight * expected_avg_search_cost_ +
            kInsertCostWeight * expected_avg_insert_cost_ * frac_inserts;
    return cost;
  }

  // Computes the expected cost of a data node constructed using the input dense
  // array of keys
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost(
      const V* values, int num_keys, double density,
      double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr, bool use_sampling = false,
      DataNodeStats* stats = nullptr) {
      if (num_keys == 0) {
        return 0;
      }

      int data_capacity =
          std::max(static_cast<int>(num_keys / density), num_keys + 1);

      // Compute what the node's model would be
      LinearModel<T> model;
      if (existing_model == nullptr) {
        build_model(values, num_keys, &model);
      } else {
        model.a_ = existing_model->a_;
        model.b_ = existing_model->b_;
      }
      model.expand(static_cast<double>(data_capacity) / num_keys);

      // Compute expected stats in order to compute the expected cost
      double cost = 0;
      double expected_avg_search_cost = 0;
      double expected_avg_insert_cost = 0;
      if (expected_insert_frac == 0) {
        ExpectedSearchCostAccumulator acc(sizeof(V), 13);
        build_node_implicit(values, num_keys, data_capacity, &acc, &model);
        expected_avg_search_cost = acc.get_stat();
      } else {
        ExpectedInsertAndSearchCostAccumulator acc(sizeof(V), 13);
        build_node_implicit(values, num_keys, data_capacity, &acc, &model);
        expected_avg_search_cost =
            acc.get_expected_search_cost();
        expected_avg_insert_cost = acc.get_expected_insert_cost();
      }
      cost = kSearchCostWeight * expected_avg_search_cost +
            kInsertCostWeight * expected_avg_insert_cost * expected_insert_frac;

      if (stats) {
        stats->num_search_cost = expected_avg_search_cost;
        stats->num_insert_cost = expected_avg_insert_cost;
      }
      return cost;
  }

  // return the overflow ratio of this punch of data
  double compute_overflow_frac(const V* values, int num_keys, int data_capacity,
    const LinearModel<T>* existing_model = nullptr) const{
    if (num_keys == 0) {
      return 0;
    }

    // Compute what the node's model would be
    LinearModel<T> model;
    if (existing_model == nullptr) {
      build_model(values, num_keys, &model);
    } else {
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }
    model.expand(static_cast<double>(data_capacity) / num_keys);

    int last_position = -1;
    int overflow_count = 0;
    for (int i = 0; i < num_keys; i++) {
      int predicted_position = std::max(
          0, std::min(data_capacity - 1, model.predict(values[i].first)));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      if((actual_position - predicted_position) >= PROBING_LENGTH){
        overflow_count++;
      }else{
        last_position = actual_position;
      }
    }
    
    return overflow_count / static_cast<double>(num_keys); 
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit(const V* values, int num_keys,
                                  int data_capacity, StatAccumulator* acc,
                                  const LinearModel<T>* model) {
    int last_position = -1;
    int last_overflow_position = -1;
    int num_in_last_overflow_blocks = 0;

    for (int i = 0; i < num_keys; i++) {
      int predicted_position = std::max(
          0, std::min(data_capacity - 1, model->predict(values[i].first)));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      if((actual_position - predicted_position) >= PROBING_LENGTH){
        // This V should be added to overflow block
        int overflow_idx = predicted_position / TABLE_FACTOR;
        if(overflow_idx != last_overflow_position){
          // Move to a new overflow position
          last_overflow_position = overflow_idx;
          num_in_last_overflow_blocks = 1;
        }else{
          num_in_last_overflow_blocks++;
        }
        acc->overflow_accumulate(num_in_last_overflow_blocks);
      }else{
        acc->accumulate(actual_position, predicted_position);
        last_position = actual_position;
      }
    }
  } 

  // Computes the expected cost of a data node constructed using the keys
  // between left and right in the
  // key/data_slots of an existing node
  // Assumes existing_model is trained on the dense array of keys
  // BT: only used in 
  static double compute_expected_cost_from_existing(
      self_type* node, int left, int right, double density,
      double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr,
      DataNodeStats* stats = nullptr,
      bool rebuild_sorted_node = false) {
    assert(left >= 0 && right <= node->data_capacity_);
    if(rebuild_sorted_node) node->build_sorted_slots();
    LinearModel<T> model;
    int num_actual_keys = 0;
    if (existing_model == nullptr) {
      order_iterator_type it(node, left, true);
      LinearModelBuilder<T> builder(&model);
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      num_actual_keys = node->num_keys_in_range(left, right, true);
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }

    if (num_actual_keys == 0) {
      return 0;
    }
    int data_capacity = std::max(static_cast<int>(num_actual_keys / density),
                                 num_actual_keys + 1);
    model.expand(static_cast<double>(data_capacity) / num_actual_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_search_cost = 0;
    double expected_avg_insert_cost = 0;
    if (expected_insert_frac == 0) {
      ExpectedSearchCostAccumulator acc(sizeof(V), 13);
      build_node_implicit_from_existing(node, left, right, num_actual_keys,
                                        data_capacity, &acc, &model);
      expected_avg_search_cost = acc.get_stat();
    } else {
      ExpectedInsertAndSearchCostAccumulator acc(sizeof(V), 13);
      build_node_implicit_from_existing(node, left, right, num_actual_keys,
                                        data_capacity, &acc, &model);      
      expected_avg_search_cost =
          acc.get_expected_search_cost();
      expected_avg_insert_cost = acc.get_expected_insert_cost();
    }
    cost = kSearchCostWeight * expected_avg_search_cost +
          kInsertCostWeight * expected_avg_insert_cost * expected_insert_frac;

    if (stats) {
      stats->num_search_cost = expected_avg_search_cost;
      stats->num_insert_cost = expected_avg_insert_cost;
    }
    return cost;
}

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit_from_existing(self_type* node, int left,
                                                int right, int num_actual_keys,
                                                int data_capacity,
                                                StatAccumulator* acc,
                                                const LinearModel<T>* model,
      bool rebuild_sorted_node = false) {
      if(rebuild_sorted_node) node->build_sorted_slots();
      int last_position = -1;
      int last_overflow_position = -1;
      int num_in_last_overflow_blocks = 0;
      order_iterator_type it(node, left, true);
      for (; it.cur_idx_ < right && !it.is_end(); it++) {
        int predicted_position =
            std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
        int actual_position =
            std::max<int>(predicted_position, last_position + 1);
        if((actual_position - predicted_position) >= PROBING_LENGTH){
          // This V should be added to overflow block
          int overflow_idx = predicted_position / TABLE_FACTOR;
          if(overflow_idx != last_overflow_position){
            // Move to a new overflow position
            last_overflow_position = overflow_idx;
            num_in_last_overflow_blocks = 1;
          }else{
            num_in_last_overflow_blocks++;
          }
          acc->overflow_accumulate(num_in_last_overflow_blocks);
        }else{
          acc->accumulate(actual_position, predicted_position);
          last_position = actual_position;
        }
      }
  }

  /*** Bulk loading and model building ***/

  // Initalize key/payload/bitmap arrays and relevant metadata
  // stash_frac >= 0.05 && stash_frac < 0.3
  void initialize(int num_keys, double density, double stash_frac = 0.05) {
    num_keys_ = num_keys;
    data_capacity_ =
        std::max(static_cast<int>(num_keys / density), num_keys + 1);
    data_capacity_ = std::max(data_capacity_, PROBING_LENGTH);

    stash_capacity_ = std::max(static_cast<int>(data_capacity_ * stash_frac), 1);
    data_capacity_ -= stash_capacity_;
    scale_factor_ = static_cast<int>(SCALE_FACTOR * (data_capacity_ / static_cast<double>(stash_capacity_ + data_capacity_))) & parameter_mask;
    if(scale_factor_ == 0) scale_factor_ = PROBING_LENGTH;
    real_data_capacity_ = data_capacity_ + PROBING_LENGTH - 1; // real data capacity in main array
  
    int metainfo_num = real_data_capacity_ / TABLE_FACTOR + 1;
    scale_parameters_size_ = data_capacity_ / scale_factor_ + (((data_capacity_ % scale_factor_) == 0) ? 0 : 1); 
    if((real_data_capacity_ / scale_factor_) > (scale_parameters_size_ - 1)){
      scale_parameters_size_++;
    }
    bitmap_size_ = static_cast<size_t>(std::ceil(stash_capacity_ / 64.));
    align_zalloc((void**)&bitmap_, sizeof(uint64_t) * bitmap_size_);
    int slot_bits = stash_capacity_ % 64;
    if (slot_bits != 0)
    {
      bitmap_[bitmap_size_ - 1] = ~((1ULL << slot_bits) - 1);
    }

    align_zalloc((void**)&scale_parameters_, scale_parameters_size_ * sizeof(ScaleParameter));
    align_zalloc((void**)&meta_info_array_, sizeof(MetaInfo) * (metainfo_num));

    int total_capacity = real_data_capacity_ + stash_capacity_;
    my_alloc::BasePMPool::Allocate(&P_data_slots_, sizeof(V) * total_capacity);
    data_slots_ = reinterpret_cast<V*>(pmemobj_direct(P_data_slots_));
    stash_slots_ = data_slots_ + real_data_capacity_;
    for(int i = 0; i < total_capacity; ++i){
      data_slots_[i].first = invalid_key_;
    }
  }

  void set_invalid_key(T invalid_key){
    invalid_key_ = invalid_key;
    // First PA
    int meta_size = real_data_capacity_ / TABLE_FACTOR + 1;
    uint32_t cur_bitmap_data;
    int cur_idx_;
    for(int i = 0; i < meta_size; ++i){
      cur_bitmap_data = ~(meta_info_array_[i].get_bitmap());
      // First part => scan the corresdnpoding data array
      while(cur_bitmap_data){
        uint32_t bit = extract_rightmost_one(cur_bitmap_data);
        cur_idx_ = i * TABLE_FACTOR + count_ones(bit - 1);
        cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
        if(cur_idx_ < real_data_capacity_){
          data_slots_[cur_idx_].first = invalid_key_;
         }
      }
    }

    my_alloc::BasePMPool::Persist(data_slots_, sizeof(V) * real_data_capacity_);

    // Then SA
    uint64_t curBitmapData;
    bool touch_limit = false;
    // Search from curBitmapIdx to (bitmap_size_ - 1)
    for(int i = 0; i < bitmap_size_; ++i){
      curBitmapData = ~(bitmap_[i]);
      while(curBitmapData != 0){
        uint64_t bit = extract_rightmost_one(curBitmapData);
        curBitmapData = curBitmapData & (~bit);
        int stash_offset = get_offset(i, bit);
        if(stash_offset >= stash_capacity_){
          touch_limit = true;
          break;
        }
        stash_slots_[stash_offset].first = invalid_key_;
      }
      if(touch_limit){
       break;
      }
    }
    my_alloc::BasePMPool::Persist(stash_slots_, sizeof(V) * stash_capacity_);

    // lastly EA
    for(int j = 0; j < scale_parameters_size_; ++j){
      index_overflow_stash_type *index_overflow_stash = scale_parameters_[j].index_overflow_stash_;
      while(index_overflow_stash){
        uint16_t bitmap;
        for(int i = 0; i < 5; ++i){
          bitmap = index_overflow_stash->bitmap_[i];
          if(bitmap & bitmapSet){
            // Search in extended stash 
            uint32_t my_bitmap = static_cast<uint32_t>(bitmap & (~bitmapSet));
            my_bitmap = (~my_bitmap) & ((1U << 16) - 1);
            while(my_bitmap){
              uint32_t bit = extract_rightmost_one(my_bitmap);
              int offset = count_ones(bit - 1);
              overflow_stash_type *overflow_stash = index_overflow_stash->overflow_stash_array_[i];
              overflow_stash->overflow_slots_[offset].first = invalid_key_;
              my_bitmap = my_bitmap & (~bit);
            }

            my_alloc::BasePMPool::Persist(index_overflow_stash->overflow_stash_array_[i], sizeof(overflow_stash_type));
          }
        }
        index_overflow_stash = index_overflow_stash->next_;
      }
    }
  }

  // Assumes pretrained_model is trained on dense array of keys
  void bulk_load(const V values[], int num_keys,
                 const LinearModel<T>* pretrained_model = nullptr,
                 bool train_with_sample = false, double stash_frac = 0.05) {
    initialize(num_keys, kInitDensity_, stash_frac);

    if (num_keys == 0) {
      expansion_threshold_ = data_capacity_;
      scale_parameters_[0].expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      return;
    }

    // Build model
    if (pretrained_model != nullptr) {
      this->model_.a_ = pretrained_model->a_;
      this->model_.b_ = pretrained_model->b_;
    } else {
      build_model(values, num_keys, &(this->model_), train_with_sample);
    }
    this->model_.expand(static_cast<double>(data_capacity_) / num_keys);

    // Model-based insert using linked list method
    for(int i = 0; i < num_keys; ++i){
      insert_to_the_node(values[i].first, values[i].second);
    }

    int total_capacity = data_capacity_ + stash_capacity_;
    expansion_threshold_ = std::min(std::max(total_capacity * kMaxDensity_,
                                             static_cast<double>(num_keys + 1)),
                                    static_cast<double>(total_capacity));
    int initial_parameter_size = data_capacity_ / scale_factor_; 
    for(int i = 0 ; i < initial_parameter_size; ++i){
      scale_parameters_[i].expansion_threshold_ = static_cast<int>(SCALE_FACTOR * kMaxDensity_);
    }

    if ((data_capacity_ % scale_factor_) != 0)
    {
      scale_parameters_[initial_parameter_size].expansion_threshold_ = std::min(std::max(static_cast<int>((total_capacity % SCALE_FACTOR) * kMaxDensity_),
                                             num_keys + 1), total_capacity);
    }
    contraction_threshold_ = total_capacity * kMinDensity_;
    max_key_ = values[num_keys - 1].first;
    min_key_ = values[0].first;

    my_alloc::BasePMPool::Persist(data_slots_, sizeof(V)*(total_capacity + PROBING_LENGTH - 1));
    my_alloc::BasePMPool::Persist(this, sizeof(self_type));
  }

  // Bulk load using the keys between the left and right positions in
  // key/data_slots of an existing data node
  // keep_left and keep_right are set if the existing node was append-mostly
  // If the linear model and num_actual_keys have been precomputed, we can avoid
  // redundant work
  void bulk_load_from_existing(
      self_type* node, int left, int right, bool keep_left = false,
      bool keep_right = false,
      const LinearModel<T>* precomputed_model = nullptr,
      int precomputed_num_actual_keys = -1, bool rebuild_sorted_node = false, double stash_frac = 0.05) {
      assert(left >= 0 && right <= node->num_keys_);
      if(rebuild_sorted_node){
        node->build_sorted_slots();
      }

      // Build model
      int num_actual_keys = 0;
      if (precomputed_model == nullptr || precomputed_num_actual_keys == -1) {
        order_iterator_type it(node, left, true);
        LinearModelBuilder<T> builder(&(this->model_));
        for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
          builder.add(it.key(), i);
          num_actual_keys++;
        }
        builder.build();
      } else {
        num_actual_keys = precomputed_num_actual_keys;
        this->model_.a_ = precomputed_model->a_;
        this->model_.b_ = precomputed_model->b_;
      }
      //BT: according to the number of keys inserted to this node, build node with target capacity
      initialize(num_actual_keys, kMinDensity_, stash_frac); // this will update num_keys_ in this node
      if (num_actual_keys == 0) {
        expansion_threshold_ = data_capacity_;
        scale_parameters_[0].expansion_threshold_ = data_capacity_;
        contraction_threshold_ = 0;
        my_alloc::BasePMPool::Persist(&this->model_, sizeof(this->model_));
        return;
      }

      // Special casing if existing node was append-mostly
      if (keep_left) {
        this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_); //BT: FIXME, this should be num_keys_in_array / KMaxDensity?
      } else if (keep_right) {
        this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
        this->model_.b_ += (data_capacity_ - (num_actual_keys / kMaxDensity_));
      } else {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
      }

      my_alloc::BasePMPool::Persist(&this->model_, sizeof(this->model_));

      order_iterator_type it(node, left, true);
      for (; it.cur_idx_ < right && !it.is_end(); it++) {
        insert_to_the_node(it.key(), it.payload());
      }

      max_key_ = node->sorted_slots_[left].first;
      min_key_ = node->sorted_slots_[right - 1].first;
      int total_capacity = data_capacity_ + stash_capacity_;
      expansion_threshold_ =
          std::min(std::max(total_capacity * kMaxDensity_,
                            static_cast<double>(num_keys_ + 1)),
                  static_cast<double>(total_capacity));
     
      int initial_parameter_size = data_capacity_ / scale_factor_; 
      for(int i = 0 ; i < initial_parameter_size; ++i){
        scale_parameters_[i].expansion_threshold_ = static_cast<int>(SCALE_FACTOR * kMaxDensity_);
      }

      if ((data_capacity_ % scale_factor_) != 0)
      {
        scale_parameters_[initial_parameter_size].expansion_threshold_ = std::min(std::max(static_cast<int>((total_capacity % SCALE_FACTOR) * kMaxDensity_),
                                              num_keys_ + 1), total_capacity);
      }

      contraction_threshold_ = total_capacity * kMinDensity_;
#ifdef ALEX_DATA_NODE_SEP_ARRAYS
      my_alloc::BasePMPool::Persist(key_slots_, sizeof(T) * real_data_capacity_);
      my_alloc::BasePMPool::Persist(payload_slots_, sizeof(P) * real_data_capacity_);
#else
      my_alloc::BasePMPool::Persist(data_slots_, sizeof(V) * (total_capacity + PROBING_LENGTH - 1));
#endif

      flush_all_overflow_stash(scale_parameters_, scale_parameters_size_);
      my_alloc::BasePMPool::Persist(&min_key_, sizeof(max_key_) * 2);
      my_alloc::BasePMPool::Persist(&expansion_threshold_, sizeof(T) * 2);
  }

  void OutputDatanodeInfo(){ 
    order_iterator_type it(this, 0);
    for (; !it.is_end(); it++) {
      std::cout << it.key() << " ";
    }
  } 

  static void build_model(const V* values, int num_keys, LinearModel<T>* model,
                          bool use_sampling = false) {
    if (use_sampling) {
      build_model_sampling(values, num_keys, model);
      return;
    }

    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i++) {
      builder.add(values[i].first, i);
    }
    builder.build();
  }

  // Uses progressive non-random uniform sampling to build the model
  // Progressively increases sample size until model parameters are relatively
  // stable
  static void build_model_sampling(const V* values, int num_keys,
                                   LinearModel<T>* model,
                                   bool verbose = false) {
    const static int sample_size_lower_bound = 10;
    // If slope and intercept change by less than this much between samples,
    // return
    const static double rel_change_threshold = 0.01;
    // If intercept changes by less than this much between samples, return
    const static double abs_change_threshold = 0.5;
    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys <= sample_size_lower_bound * sample_size_multiplier) {
      build_model(values, num_keys, model, false);
      return;
    }

    int step_size = 1;
    double sample_size = num_keys;
    while (sample_size >= sample_size_lower_bound) {
      sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;

    // Run with initial step size
    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i += step_size) {
      builder.add(values[i].first, i);
    }
    builder.build();
    double prev_a = model->a_;
    double prev_b = model->b_;
    if (verbose) {
      std::cout << "Build index, sample size: " << num_keys / step_size
                << " (a, b): (" << prev_a << ", " << prev_b << ")" << std::endl;
    }

    // Keep decreasing step size (increasing sample size) until model does not
    // change significantly
    while (step_size > 1) {
      step_size /= sample_size_multiplier;
      // Need to avoid processing keys we already processed in previous samples
      int i = 0;
      while (i < num_keys) {
        i += step_size;
        for (int j = 1; (j < sample_size_multiplier) && (i < num_keys);
             j++, i += step_size) {
          builder.add(values[i].first, i);
        }
      }
      builder.build();

      double rel_change_in_a = std::abs((model->a_ - prev_a) / prev_a);
      double abs_change_in_b = std::abs(model->b_ - prev_b);
      double rel_change_in_b = std::abs(abs_change_in_b / prev_b);
      if (verbose) {
        std::cout << "Build index, sample size: " << num_keys / step_size
                  << " (a, b): (" << model->a_ << ", " << model->b_ << ") ("
                  << rel_change_in_a << ", " << rel_change_in_b << ")"
                  << std::endl;
      }
      if (rel_change_in_a < rel_change_threshold &&
          (rel_change_in_b < rel_change_threshold ||
           abs_change_in_b < abs_change_threshold)) {
        return;
      }
      prev_a = model->a_;
      prev_b = model->b_;
    }
  }

  // Unused function: builds a spline model by connecting the smallest and
  // largest points instead of using
  // a linear regression
  static void build_spline(const V* values, int num_keys,
                           const LinearModel<T>* model) {
    int y_max = num_keys - 1;
    int y_min = 0;
    model->a_ = static_cast<double>(y_max - y_min) /
                (values[y_max].first - values[y_min].first);
    model->b_ = -1.0 * values[y_min].first * model->a_;
  }

  /*** Lookup ***/

  // Predicts the position of a key using the model
  inline int predict_position(const T& key) const {
    int position = this->model_.predict(key);
    position = std::max<int>(std::min<int>(position, data_capacity_ - 1), 0);
    return position;
  }

  // Fisrt return whether it need return, the second mean whether the payload is found or not
  inline bool find_payload(const T& key, P* ret, bool* found) const {
    int predicted_pos = predict_position(key);
    int parameter_pos = predicted_pos / scale_factor_;
    uint32_t version;
    if(test_lock_set(version, parameter_pos)) return false;
    
    // Return whether the payload is found or not
    *found = linear_probing_search_exact(predicted_pos, parameter_pos, key, ret);

    if(test_lock_version_change(version, parameter_pos)) return false;
    return true;
  }
  
  // Continue to add new record to result array, and shift the cur_idx
  // cur_idx points to a position that are empty
  // Return the cur_idx
  inline int add_to_sorted_result(V* result, V* new_record, int total_size, int cur_idx){
    if (cur_idx >= total_size)
    {
      // if (result[total_size - 1].first < new_record->first)
      if (key_less(result[total_size - 1].first, new_record->first))
      {
        return cur_idx;
      }
      cur_idx = total_size - 1; // Remove the last element
    }

    // Start the insertion sort
    int j = cur_idx - 1;
    //while((j >= 0) && (result[j].first > new_record->first)){
    while((j >= 0) && (key_greater(result[j].first, new_record->first))){
      result[j + 1] = result[j];
      --j;
    }

    result[j + 1] = *new_record;
    ++cur_idx;
    return cur_idx;
  }

  // Scan all records in a array
  inline void add_data_slots_to_result(const T& min_key, int meta_pos, int internal_pos, V* result, int total_size, int* cur_idx){
    uint32_t cur_bitmap_data = (meta_info_array_[meta_pos].get_bitmap() >> internal_pos) << internal_pos;
    V cur_key_payload;
    int base_pos = meta_pos * TABLE_FACTOR;

    while(cur_bitmap_data){
      uint32_t bit = extract_rightmost_one(cur_bitmap_data);
      uint32_t record_pos = base_pos + count_ones(bit - 1);
      cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
      //if(data_slots_[record_pos].first >= min_key){
      if(key_greaterequal(data_slots_[record_pos].first, min_key)){
        *cur_idx = add_to_sorted_result(result, &(data_slots_[record_pos]), total_size, *cur_idx);
      }
    }
  }
  
  // Scan all records in a link list
  inline void add_overflow_finger_to_result(const T& min_key, int meta_pos, V* result, int total_size, int* cur_idx){ 
    uint32_t cur_bitmap_data;
    V cur_key_payload;
    auto cur_overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
    while(cur_overflow_finger){
      cur_bitmap_data = cur_overflow_finger->get_bitmap();
      while(cur_bitmap_data){
        uint32_t bit = extract_rightmost_one(cur_bitmap_data);
        auto record_pos = count_ones(bit - 1);
        cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
        cur_key_payload = *(reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(cur_overflow_finger->overflow_array_[record_pos])));
        //if(cur_key_payload.first >= min_key){
        if(key_greaterequal(cur_key_payload.first, min_key)){
          *cur_idx = add_to_sorted_result(result, &cur_key_payload, total_size, *cur_idx);
        }
      }
      cur_overflow_finger = cur_overflow_finger->next_;
    }
  }

  // No need to care about the concurrency here, just return the return in a range
  // start is the pos on data array
  // end_pos is the pos on meta_info_array
  // Return whether it already conduct the safe check
  // In this fucntion, it is only managed by one parameter lock and search util the enough records are collected or the region of safe managed if touched
  // External function needs to guarantee the safeness of the result array
  inline bool range_scan_one_parameter(const T& min_key, int start, V* result, int total_size, int* cur_idx, int parameter_pos){
      int start_pos = start / TABLE_FACTOR;  
      int start_internal_pos = start & table_factor_hide;
      bool safe_check = false;
      bool need_check_version_lock = true;
      int metainfo_num = real_data_capacity_ / TABLE_FACTOR + 1;
      int end_pos = (parameter_pos + 1) * scale_factor_ / TABLE_FACTOR - 1; 
      end_pos = std::max<int>(std::min<int>(end_pos, metainfo_num - 1), 0); // aovid that the parameter_pos is the last parameter pos
      
      int org_idx = *cur_idx;
      do{
        // concurrency control
        uint32_t version;
        while(test_lock_set(version, parameter_pos)){
          if(test_lock_set(version)){
            need_check_version_lock = false;
            break;
          }
        }

        for(int i = start_pos; i <= end_pos; ++i){
          add_data_slots_to_result(min_key, i, start_internal_pos, result, total_size, cur_idx);
          add_overflow_finger_to_result(min_key, i, result, total_size, cur_idx);
          if((*cur_idx) >= total_size){
            if(i < end_pos){
              add_data_slots_to_result(min_key, i + 1, 0, result, total_size, cur_idx);
              safe_check = true;
            }
            break;
          }
          start_internal_pos = 0;
        }

        if(need_check_version_lock && test_lock_version_change(version, parameter_pos)){
          // Need retry, reset the return array
          *cur_idx = org_idx;
          safe_check = false;
        }else{
          break;
        }
    }while(true);
    return safe_check;
  }

  // Scan all records in a array
  // This also needs guarantee the safety of result array
  // Only the records that are smaller than max_key could be added to the resul array
  inline void add_data_slots_to_result_with_less(const T& min_key, const T& max_key, int meta_pos, int parameter_pos, V* result, int total_size, int* cur_idx){
     bool need_check_version_lock = true;
     thread_local V tmp[PROBING_LENGTH]; // use this to store the results
     int num_valid_data = 0;

     do{
        // concurrency control
        uint32_t version;
        while(test_lock_set(version, parameter_pos)){
          if(test_lock_set(version)){
            need_check_version_lock = false;
            break;
          }
        }

        uint32_t cur_bitmap_data = meta_info_array_[meta_pos].get_bitmap();
        V cur_key_payload;
        int base_pos = meta_pos * TABLE_FACTOR;
        while(cur_bitmap_data){
          uint32_t bit = extract_rightmost_one(cur_bitmap_data);
          uint32_t record_pos = base_pos + count_ones(bit - 1);
          cur_bitmap_data = remove_rightmost_one(cur_bitmap_data);
          //if((data_slots_[record_pos].first >= min_key) && (data_slots_[record_pos].first < max_key)){
          if((key_greaterequal(data_slots_[record_pos].first, min_key)) && (key_less(data_slots_[record_pos].first, max_key))){
            tmp[num_valid_data] = data_slots_[record_pos];
            num_valid_data++;
          }
        }

        if((!need_check_version_lock) || (!test_lock_version_change(version, parameter_pos))){
          break;
        }

        num_valid_data = 0;
    }while(true);

    // Add these data to the sorted array
    for(int i = 0; i < num_valid_data; ++i){
      *cur_idx = add_to_sorted_result(result, &(tmp[i]), total_size, *cur_idx);
    }
  }

  // Need to carefully consider the CC during range scan    
  // Use recursion for implementation
  inline int range_scan_by_size(const T& key, uint32_t to_scan, V* result, uint32_t global_version) {
    // First recover the data node 
    if(local_version_ != global_version){
      recover_node(global_version);
    }

    // T debug_key = 18744.9674616;
    int predicted_pos = predict_position(key); // First locate this position
    int parameter_pos = predicted_pos / scale_factor_; // Determine the parameter pos - start of the lock array
    int metainfo_num = real_data_capacity_ / TABLE_FACTOR + 1; // Total metainfo number
    
    bool safe_check = false; // Whether the result array has contain all "necessary (= small enough)" key
    T min_key = key; // Remaining scanning key need to be equal or larger than this key
    V* cur_result = result;
    int cur_idx = 0; // The array index of currentresult array
    int start_idx = predicted_pos; // Start from which array index
    int remaining_scan = to_scan;  // Number of remaining kv records for scanning

    // Scan from start to scale_parameters_size_ - 1
    for (int i = parameter_pos; i < scale_parameters_size_; ++i){
      safe_check = range_scan_one_parameter(min_key, start_idx, cur_result, remaining_scan, &cur_idx, i);
      if(cur_idx >= remaining_scan){ // The result array is already full
        if(!safe_check && (i != (scale_parameters_size_ - 1))){
          int meta_pos = (i + 1) * scale_factor_ / TABLE_FACTOR;
          meta_pos = std::max<int>(std::min<int>(meta_pos, metainfo_num - 1), 0);
          add_data_slots_to_result_with_less(key, cur_result[cur_idx - 1].first, meta_pos, i + 1, cur_result, remaining_scan, &cur_idx);
        }
        remaining_scan = 0;
        break;
      }

      // Not enough results, need to first do some safe collection
      if ((cur_idx != 0) && (i != (scale_parameters_size_ - 1))){
        int meta_pos = (i + 1) * scale_factor_ / TABLE_FACTOR;
        meta_pos = std::max<int>(std::min<int>(meta_pos, metainfo_num - 1), 0);
        int old_idx = cur_idx;
        add_data_slots_to_result_with_less(key, cur_result[cur_idx - 1].first, meta_pos, i + 1, cur_result, remaining_scan, &cur_idx);
        // Check whetehr it is enough
        if(cur_idx >= remaining_scan){
          remaining_scan = 0;
          break;
        }
      }

      // Use the new min key, and reset other parameters such as result array, cur_idx, to scan
      start_idx = (i + 1) * scale_factor_;
      if(cur_idx != 0){
        min_key = cur_result[cur_idx - 1].first;
        cur_result = cur_result + cur_idx;
        remaining_scan = remaining_scan - cur_idx;
        cur_idx = 0;
      }
    }

    if(remaining_scan == 0){
      return to_scan;
    }
    
    if(next_leaf_ != nullptr){ // Need to go to next node for collection
      auto scan_result = next_leaf_->range_scan_by_size(min_key, remaining_scan, cur_result, global_version); 
      remaining_scan -= scan_result;
    }
    
    return (to_scan - remaining_scan);
  }

  // start is inclusive, end is exclusive
  inline int find_free_pos_in_range(int start, int end){ 
    //new version which uses the bitmap in stash map
    int start_pos = start / TABLE_FACTOR;
    int start_internal_pos = start & table_factor_hide;
    int num_in_start = TABLE_FACTOR - start_internal_pos; //num-bits in first bitmap

    int end_pos = end / TABLE_FACTOR;
    int end_internal_pos = end & table_factor_hide;
    uint32_t my_bitmap = meta_info_array_[start_pos].get_bitmap() >> start_internal_pos;
    if(end_internal_pos != 0){
      my_bitmap |= ((meta_info_array_[end_pos].get_bitmap() & ((1U << end_internal_pos) - 1)) << num_in_start);
    }

    uint32_t bit = extract_rightmost_zero(my_bitmap);
    int pos = start + count_ones(bit - 1);
    if(pos >= end) return -1;
    return pos;
  }

  // start is inclusive, end is not inclusive
  inline int find_free_pos_in_range(int start, int end, uint32_t my_bitmap){ 
    uint32_t bit = extract_rightmost_zero(my_bitmap);
    int pos = start + count_ones(bit - 1);
    if(pos >= end) return -1;
    return pos;
    //already compose the bitmap
  }

   // start is inclusive, end is exclusive
  inline int find_free_pos_in_range(MetaInfo* new_meta_info_array, int start, int end){ 
    //new version which uses the bitmap in stash map
    int start_pos = start / TABLE_FACTOR;
    int start_internal_pos = start & table_factor_hide;
    int num_in_start = TABLE_FACTOR - start_internal_pos; //num-bits in first bitmap

    int end_pos = end / TABLE_FACTOR;
    int end_internal_pos = end & table_factor_hide;
    uint32_t my_bitmap = new_meta_info_array[start_pos].get_bitmap() >> start_internal_pos;
    if(end_internal_pos != 0){
      my_bitmap |= ((new_meta_info_array[end_pos].get_bitmap() & ((1U << end_internal_pos) - 1)) << num_in_start);
    }

    uint32_t bit = extract_rightmost_zero(my_bitmap);
    int pos = start + count_ones(bit - 1);
    if(pos >= end) return -1;
    return pos;
    //already compose the bitmap
  }

  inline void insert_overflowfp_to_meta_without_cost(int meta_pos, V* overflow_data, unsigned char fp, uint8_t offset){
    // First search a free slot
    V* attached = reinterpret_cast<V*>(reinterpret_cast<uint64_t>(overflow_data) | (static_cast<uint64_t>(fp) << 56) | (static_cast<uint64_t>(offset) << 48));
    auto overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
    while(overflow_finger != nullptr){
      for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
        if(overflow_finger->overflow_array_[i] == nullptr){
          overflow_finger->overflow_array_[i] = attached;
          return;
        }
      }
      overflow_finger = overflow_finger->next_;
    }

    // Create a new slot and attach to the meta_array
    OverflowFinger *new_overflow_finger;
    align_zalloc((void**)&new_overflow_finger, sizeof(OverflowFinger));
    new_overflow_finger->next_ = meta_info_array_[meta_pos].get_overflow_finger();
    new_overflow_finger->overflow_array_[0] = attached;
    meta_info_array_[meta_pos].set_overflow_finger(new_overflow_finger);
  }

  // this will collect cost in parameter set
  inline void insert_overflowfp_to_meta(int meta_pos, int parameter_pos, V* overflow_data, unsigned char fp, uint8_t offset){
    // First search a free slot
    V* attached = reinterpret_cast<V*>(reinterpret_cast<uint64_t>(overflow_data) | (static_cast<uint64_t>(fp) << 56) | (static_cast<uint64_t>(offset) << 48));
    auto overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
    int overflow_items = 0;
    while(overflow_finger != nullptr){
      for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
        if(overflow_finger->overflow_array_[i] == nullptr){
          overflow_items += i;
          overflow_finger->overflow_array_[i] = attached;
          scale_parameters_[parameter_pos].num_search_cost_ +=  sizeof(V) * (overflow_items + PROBING_LENGTH) / 64.0 + 1;
          scale_parameters_[parameter_pos].num_insert_cost_ += 0.5;
          return;
        }
      }
      overflow_items += OVERFLOW_FINGER_LENGTH;
      overflow_finger = overflow_finger->next_;
    }

    // Create a new slot and attach to the meta_array
    OverflowFinger *new_overflow_finger;
    align_zalloc((void**)&new_overflow_finger, sizeof(OverflowFinger));
    new_overflow_finger->next_ = meta_info_array_[meta_pos].get_overflow_finger();
    new_overflow_finger->overflow_array_[0] = attached;
    meta_info_array_[meta_pos].set_overflow_finger(new_overflow_finger);
    scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * (overflow_items + PROBING_LENGTH) / 64.0 + 1;
    scale_parameters_[parameter_pos].num_insert_cost_ += 2.5; // Allocate a new overflow node
  }

  // Following functions are all about how to insert KV to overflow stash
  // Need to allocate additional persistent memory => to ensure crash-consistency during PM allocation
  // NO synchronization in this function, so it is only invoked in SMO
  inline V* insert_to_overflow_stash_without_cost(int parameter_pos, int* ret_offset, const T& key, const P& payload){
    index_overflow_stash_type *index_overflow_stash = scale_parameters_[parameter_pos].index_overflow_stash_;
    while(index_overflow_stash){
      auto ret = index_overflow_stash_insert_kv(index_overflow_stash, key, payload, ret_offset);
      if (ret != nullptr)
      {
        return ret;
      }

      if((*ret_offset) != -1){
        V* insert_v;
        overflow_stash_type *new_overflow_stash;
        if (scale_parameters_[parameter_pos].overflow_stash_count_ == 0)
        {
          insert_v = index_overflow_stash_insert_alloc(index_overflow_stash, key, payload, invalid_key_, &P_data_slots_, parameter_pos, *ret_offset);
          //overflow_stash_type::New(&P_data_slots_, key, payload, invalid_key_, parameter_pos);
          new_overflow_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(P_data_slots_));
          new_overflow_stash->link_ = first_block_;
          clwb(&(new_overflow_stash->link_));
          sfence();
          first_block_ = new_overflow_stash;
          clwb(&first_block_);
          sfence();
        }else{
          auto last_stash = scale_parameters_[parameter_pos].last_stash_;
          insert_v = index_overflow_stash_insert_alloc(index_overflow_stash, key, payload, invalid_key_, &(last_stash->next_), parameter_pos, *ret_offset);
          new_overflow_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(last_stash->next_));
        }
        scale_parameters_[parameter_pos].last_stash_ = new_overflow_stash;
        scale_parameters_[parameter_pos].overflow_stash_count_++;
        *ret_offset = 0;
        return insert_v;
      }

      index_overflow_stash = index_overflow_stash->next_;
    }

    *ret_offset = 0;
    // Create a new index overflow
    index_overflow_stash_type *new_index;
    align_zalloc((void**)(&new_index), sizeof(index_overflow_stash_type));
    overflow_stash_type *new_overflow_stash;
    if (scale_parameters_[parameter_pos].overflow_stash_count_ == 0)
    {
      overflow_stash_type::New(&P_data_slots_, key, payload, invalid_key_, parameter_pos);
      new_overflow_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(P_data_slots_));
      new_overflow_stash->link_ = first_block_;
      clwb(&(new_overflow_stash->link_));
      sfence();
      first_block_ = new_overflow_stash;
      clwb(&first_block_);
      sfence();
    }else{
      auto last_stash = scale_parameters_[parameter_pos].last_stash_;
      overflow_stash_type::New(&(last_stash->next_), key, payload, invalid_key_, parameter_pos);
      new_overflow_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(last_stash->next_));
    }
    scale_parameters_[parameter_pos].last_stash_ = new_overflow_stash;
    scale_parameters_[parameter_pos].overflow_stash_count_++;
    new_index->overflow_stash_array_[0] = new_overflow_stash;
    new_index->bitmap_[0] = ((uint16_t)1 << 15) | (uint16_t)1;
  
    // attach overflow stash to node
    new_index->next_ = scale_parameters_[parameter_pos].index_overflow_stash_;
    scale_parameters_[parameter_pos].index_overflow_stash_ = new_index;
    return new_overflow_stash->overflow_slots_;
  }

  inline void link_overflow_stash_with_sync(int parameter_pos, overflow_stash_type *new_overflow_stash){
    overflow_stash_type* old_value = first_block_;
    do{
      new_overflow_stash->link_ = old_value;
      clwb(&(new_overflow_stash->link_));
    }while(!CAS(&first_block_, &old_value, new_overflow_stash));
    clwb(&first_block_);
    sfence();
  }

  inline V*  insert_to_overflow_stash_with_sync(int parameter_pos, int *ret_offset, const T& key, const P& payload, OverflowDesc *desc){
    index_overflow_stash_type *index_overflow_stash = scale_parameters_[parameter_pos].index_overflow_stash_;
    while(index_overflow_stash){
      auto ret = index_overflow_stash_insert_kv(index_overflow_stash, key, payload, ret_offset);
      if (ret != nullptr)
      {
        return ret;
      }

      if((*ret_offset) != -1){ // used to allocate another overflow stash block
        V* insert_v;
        overflow_stash_type *new_overflow_stash;
        if (scale_parameters_[parameter_pos].overflow_stash_count_ == 0)
        {
          desc->new_node_ = OID_NULL;
          desc->cur_node_ = reinterpret_cast<void*>(this);
          insert_v = index_overflow_stash_insert_alloc(index_overflow_stash, key, payload, invalid_key_, &(desc->new_node_), parameter_pos, *ret_offset);
          new_overflow_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(desc->new_node_));
          link_overflow_stash_with_sync(parameter_pos, new_overflow_stash);
          desc->cur_node_ = nullptr;
          clwb(&desc->cur_node_);
          sfence();
        }else{
          auto last_stash = scale_parameters_[parameter_pos].last_stash_;
          insert_v = index_overflow_stash_insert_alloc(index_overflow_stash, key, payload, invalid_key_, &(last_stash->next_), parameter_pos, *ret_offset);
          new_overflow_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(last_stash->next_));
        }
        scale_parameters_[parameter_pos].last_stash_ = new_overflow_stash;
        scale_parameters_[parameter_pos].overflow_stash_count_++;
        *ret_offset = 0;
        return insert_v;
      }

      index_overflow_stash = index_overflow_stash->next_;
    }

    // Create a new index overflow
    index_overflow_stash_type *new_index;
    align_zalloc((void**)(&new_index), sizeof(index_overflow_stash_type));
    overflow_stash_type *new_stash;
    V* insert_v;
    if (scale_parameters_[parameter_pos].overflow_stash_count_ == 0)
    {
      desc->new_node_ = OID_NULL;
      desc->cur_node_ = reinterpret_cast<void*>(this);
      insert_v = index_overflow_stash_insert_alloc(new_index, key, payload, invalid_key_, &(desc->new_node_), parameter_pos, 0);
      new_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(desc->new_node_));
      link_overflow_stash_with_sync(parameter_pos, new_stash);
      desc->cur_node_ = nullptr;
      clwb(&desc->cur_node_);
      sfence();
    }else{
      auto last_stash = scale_parameters_[parameter_pos].last_stash_;
      insert_v = index_overflow_stash_insert_alloc(new_index, key, payload, invalid_key_, &(last_stash->next_), parameter_pos, 0);
      new_stash = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(last_stash->next_));
    }

    scale_parameters_[parameter_pos].last_stash_ = new_stash;
    scale_parameters_[parameter_pos].overflow_stash_count_++;
    new_index->next_ = scale_parameters_[parameter_pos].index_overflow_stash_;
    scale_parameters_[parameter_pos].index_overflow_stash_ = new_index;
    *ret_offset = 0;
    return insert_v;
  }

  bool is_overflow_stash_linked(overflow_stash_type *stash_node){
    overflow_stash_type *cur_node = first_block_;
    while(cur_node){
      if(cur_node == stash_node) return true;
      cur_node = cur_node->link_;
    }
    return false;
  }

  inline bool erase_from_overflow_stash(int parameter_pos, overflow_stash_type* target_stash, int offset){
    index_overflow_stash_type *index_overflow_stash = scale_parameters_[parameter_pos].index_overflow_stash_;
    while(index_overflow_stash){
      uint16_t bitmap;
      V* ret = nullptr;
      for(int i = 0; i < 5; ++i){
        bitmap = index_overflow_stash->bitmap_[i];
        if((bitmap & bitmapSet) && (index_overflow_stash->overflow_stash_array_[i] == target_stash)){
          index_overflow_stash->bitmap_[i] &= ~((uint16_t)1 << offset);
          return true;
        }
      }
      index_overflow_stash = index_overflow_stash->next_;
    }
    return false;
  }

  // This function has no synchronization, it is used in bulk loading and SMO
  inline void insert_to_stash_without_cost(int predicted_pos, const T& key, const P& payload, unsigned char fp){ 
    // First find a free slot in stash area
    int stash_pos = find_free_pos_in_stash(predicted_pos);
    V *overflow_data = nullptr;
    uint8_t offset = 0;
    if(stash_pos < 0){
      // No space left in stash area, find a slot in scale_parameter
      int parameter_pos = predicted_pos / scale_factor_;
      int insert_offset;
      overflow_data = insert_to_overflow_stash_without_cost(parameter_pos, &insert_offset, key, payload);
      offset = offsetSet | static_cast<uint8_t>(insert_offset);
    }else{
      stash_slots_[stash_pos] = std::make_pair(key, payload);
      overflow_data = stash_slots_ + stash_pos;
    }

    int meta_pos = predicted_pos / TABLE_FACTOR;
    // prepare the overflow fingperprint
    insert_overflowfp_to_meta_without_cost(meta_pos, overflow_data, fp, offset);
  }

  inline void insert_to_stash_with_sync(int predicted_pos, int parameter_pos, int meta_pos, const T& key, const P& payload, unsigned char fp, OverflowDesc *overflow_desc){
    // First find a free slot in stash area
    int stash_pos = find_free_pos_in_stash_with_sync(predicted_pos);
    V *overflow_data = nullptr;
    uint8_t offset = 0;
    if(stash_pos < 0){     
      // No space left in stash area, find a slot in scale_parameter
      int insert_offset = 0;
      overflow_data = insert_to_overflow_stash_with_sync(parameter_pos, &insert_offset, key, payload, overflow_desc);
      offset = offsetSet | static_cast<uint8_t>(insert_offset);
    }else{
      stash_slots_[stash_pos].second = payload;
      stash_slots_[stash_pos].first = key;
      overflow_data = stash_slots_ + stash_pos;
    } 

    clwb(overflow_data);
    sfence();
    // Insert the stash info to DRAM-stored OverflowFinger
    insert_overflowfp_to_meta(meta_pos, parameter_pos, overflow_data, fp, offset);
  }

  void flush_all_overflow_stash(ScaleParameter* scale_parameters, size_t scale_parameters_size){
    uint16_t bitmap;
    for(int i = 0; i < scale_parameters_size; ++i){
      index_overflow_stash_type *index_overflow_stash = scale_parameters[i].index_overflow_stash_;
      while(index_overflow_stash){
        for(int i = 0; i < 5; ++i){
          bitmap = index_overflow_stash->bitmap_[i];
          if(bitmap & bitmapSet){
            overflow_stash_type *overflow_stash = index_overflow_stash->overflow_stash_array_[i];
            my_alloc::BasePMPool::Persist(overflow_stash, sizeof(overflow_stash_type));
          }
        }
        index_overflow_stash = index_overflow_stash->next_;
      }
    }
  }

 std::pair<int, int> stash_block_num(ScaleParameter* scale_parameters, int scale_parameters_size){
    uint16_t bitmap;
    int block_count = 0;
    int key_count = 0;
    for(int i = 0; i < scale_parameters_size; ++i){
      index_overflow_stash_type *index_overflow_stash = scale_parameters[i].index_overflow_stash_;
      while(index_overflow_stash){
        for(int i = 0; i < 5; ++i){
          bitmap = index_overflow_stash->bitmap_[i];
          if(bitmap & bitmapSet){
            overflow_stash_type *overflow_stash = index_overflow_stash->overflow_stash_array_[i];
            ++block_count;
            key_count += count_ones(static_cast<uint32_t>((bitmap & (~bitmapSet))));
          }
        }
        index_overflow_stash = index_overflow_stash->next_;
      }
    }
    return {block_count, key_count};
  }


  void deallocate_all_overflow_stash(ScaleParameter* scale_parameters, size_t scale_parameters_size){
    uint16_t bitmap;
    for(int i = 0; i < scale_parameters_size; ++i){
      index_overflow_stash_type *index_overflow_stash = scale_parameters[i].index_overflow_stash_;
      while(index_overflow_stash){
        for(int i = 0; i < 5; ++i){
          bitmap = index_overflow_stash->bitmap_[i];
          if(bitmap & bitmapSet){
            overflow_stash_type *overflow_stash = index_overflow_stash->overflow_stash_array_[i];
            auto ptr = pmemobj_oid(overflow_stash);
            pmemobj_free(&ptr);
          }
        }
        auto old_stash = index_overflow_stash;
        index_overflow_stash = index_overflow_stash->next_;
        free(old_stash);
      }
    }
  }

  void deallocate_all_overflow_finger(MetaInfo* meta_info_array, int meta_capacity){
    OverflowFinger *cur_overflow_finger;
    for(int i = 0; i < meta_capacity; ++i){
      cur_overflow_finger = meta_info_array[i].get_overflow_finger();
      while(cur_overflow_finger){
        auto next_overflow_finger = cur_overflow_finger->next_;
        free(cur_overflow_finger);
        cur_overflow_finger = next_overflow_finger;
      }
    }
  }

  void insert_or_increment(std::map<int, int>& my_map, int key, int incre_value){
    auto iter = my_map.find(key);
    if(iter != my_map.end()){
      iter->second += incre_value;
    }else{
      my_map[key] = incre_value;
    }
  }

  void collect_overflow_length_and_items(std::map<int, int>& length_map, std::map<int, int>& number_map, std::map<int, int>& ratio_map, int& meta_array_length, int& filled_array_length, int& total_overflow_items){
    OverflowFinger *cur_overflow_block = nullptr;
    auto meta_size = real_data_capacity_ / TABLE_FACTOR + 1;
    meta_array_length += meta_size;

    int filled_items = 0;
    for(int i = 0; i < meta_size; ++i){
      if(meta_info_array_[i].get_overflow_finger() != nullptr){
        filled_items++;
        int num_blocks = 0;
        int num_keys = 0;
        cur_overflow_block = meta_info_array_[i].get_overflow_finger();
        while(cur_overflow_block != nullptr){
          num_blocks++;
          num_keys += count_ones(cur_overflow_block->get_bitmap());
          cur_overflow_block = cur_overflow_block->next_;
        }
        total_overflow_items += num_keys;
        insert_or_increment(length_map, num_blocks, 1);
        insert_or_increment(number_map,num_keys, 1);
        filled_array_length++;
      }
    }
    if(filled_items != 0){
      int ratio = meta_size / filled_items;
      insert_or_increment(ratio_map, ratio, 1);
    }
  }

  // first int the scale_parameter size, second is the number of overflow blocks
  std::pair<int, int> collect_overflow_stash_blocks(){
    int num = 0;
    for(int i = 0; i < scale_parameters_size_; ++i){
      index_overflow_stash_type *index_overflow_stash = scale_parameters_[i].index_overflow_stash_;
      while(index_overflow_stash){
        for (int j = 0; j < 5; ++j)
        {
           if(index_overflow_stash->bitmap_[j] & bitmapSet){
              ++num;
            }
        }
        index_overflow_stash = index_overflow_stash->next_;
      }
    }
    return {scale_parameters_size_, num};
  }


  // Starting from a position, return the first position that is not a gap
  // If no more filled positions, will return data_capacity
  // If exclusive is true, output is at least (pos + 1)
  // If exclusive is false, output can be pos itself
  int get_next_filled_position(int pos, bool exclusive) const {
    printf("This function should not be clalled!!\n");
    return -1;
  }


  // Return whether the payload is found or not
  template<class K>
  inline bool linear_probing_search_exact(int m, int parameter_pos, const K& key, P* payload) const {
    scale_parameters_[parameter_pos].num_lookups_++;
    int end = m + PROBING_LENGTH;
    for(int i = m; i < end; ++i){
      if(key_equal(ALEX_DATA_NODE_KEY_AT(i), key)){
          scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * (i - m) / 64.0;   
          *payload = ALEX_DATA_NODE_PAYLOAD_AT(i);    
          return true;
      }
    }

    // Search in overflow FP
    unsigned char fp = hashcode1B<K>(key);
    int meta_pos = m / TABLE_FACTOR;
    OverflowFinger *overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
    int overflow_count = 0;
    while(overflow_finger != nullptr){
      V** overflow_array = overflow_finger->overflow_array_;
      for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
        if((overflow_array[i] != nullptr) && (static_cast<uint8_t>(reinterpret_cast<uint64_t>(overflow_array[i]) >> 56) == fp)){
          V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
          if(key_equal(ret->first, key)){
              overflow_count += i;
              scale_parameters_[parameter_pos].num_search_cost_ += 4 + sizeof(V) * overflow_count / 64.0;  
              *payload = ret->second;    
              return true;
          }
        }
      }
      overflow_count += OVERFLOW_FINGER_LENGTH;
      overflow_finger = overflow_finger->next_;
    }

    return false;
  }

      template<class K>
  bool linear_probing_negative_search(int m, const K& key, unsigned char fp, uint32_t& compose_bitmap){
    uint32_t my_bitmap = 0; //composed bitmap
    char my_fp[16]; //composed fingerprint array

    int start_pos = m / TABLE_FACTOR;
    int start_internal_pos = m & table_factor_hide;
    int num_in_start =  TABLE_FACTOR - start_internal_pos;

    int end_pos = (m + PROBING_LENGTH) / TABLE_FACTOR;
    int end_internal_pos = (m + PROBING_LENGTH) & table_factor_hide; 

    // Handle the first case
    memcpy(my_fp, &meta_info_array_[start_pos].fingerprint_[start_internal_pos], num_in_start);
    my_bitmap = meta_info_array_[start_pos].get_bitmap() >> start_internal_pos;
    //Second case
    if(end_internal_pos != 0){
      memcpy(my_fp + num_in_start, &meta_info_array_[end_pos].fingerprint_[0], end_internal_pos);
      my_bitmap |= ((meta_info_array_[end_pos].get_bitmap() & ((1U << end_internal_pos) - 1)) << num_in_start);
    }

    uint32_t mask;
    SSE_CMP8(my_fp, fp);
    compose_bitmap = my_bitmap;
    mask &= my_bitmap;

    // search every matching candidate
    while (mask) {
        //wrong_positive_search_times++;
        int jj = bitScan(mask)-1;  // next candidate
        if (key_equal(ALEX_DATA_NODE_KEY_AT(m + jj), key)) { // found: do nothing, return
           return true;
        }
        mask &= ~(0x1<<jj);  // remove this bit
    } // end while

    OverflowFinger *overflow_finger = meta_info_array_[start_pos].get_overflow_finger();
    while(overflow_finger != nullptr){
      V** overflow_array = overflow_finger->overflow_array_;
      for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
        if((overflow_array[i] != nullptr) && (static_cast<uint8_t>(reinterpret_cast<uint64_t>(overflow_array[i]) >> 56) == fp)){
          V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
          if(key_equal(ret->first, key)){
              return true;   
          }
        }
      }
      overflow_finger = overflow_finger->next_;
    }

    // Not exist, thus return nullptr
    return false;
  }

   // Searches for the first position no less than key
  // This could be the position for a gap (i.e., its bit in the bitmap is 0)
  // Returns position in range [0, data_capacity]
  // Compare with find_lower()
  template <class K>
  int lower_bound(const K& key, bool use_sorted_node = false, bool build_sorted_node = false) {
      if(use_sorted_node == false) std::cout << "Wrong condition in lower bound" << std::endl;
      if(build_sorted_node) build_sorted_slots(); 
      for(int i = 0; i < num_keys_; ++i){
        if(key_greaterequal(sorted_slots_[i].first, key)) return i;
      }
      return num_keys_;
  }


  /*** Inserts and resizes ***/

  // Whether empirical cost deviates significantly from expected cost
  // Also returns false if empirical cost is sufficiently low and is not worth
  // splitting
  inline bool significant_cost_deviation() const {
    //BT: MODEL FIX
  #ifdef NEW_COST_MODEL
    double emp_cost = new_empirical_cost();
  #else  
    double emp_cost = empirical_cost();
  #endif  
    return data_capacity_ > 120 && emp_cost > 2 * kNodeLookupsWeight && emp_cost > 2 * this->cost_;
  }

  // Returns true if cost is catastrophically high and we want to force a split
  // The heuristic for this is if the number of shifts per insert (expected or
  // empirical) is over 100
  inline bool catastrophic_cost() const {
    // BT: MODEL FIX
#ifdef NEW_COST_MODEL
    return  insert_cost_per_insert() > 25 || expected_avg_insert_cost_ > 25;
#else
    return shifts_per_insert() > 100 || expected_avg_shifts_ > 100;
#endif
  }

  inline void set_fingerprint(unsigned char fp, int pos){
    int array_pos = pos / TABLE_FACTOR;
    int internal_pos = pos & table_factor_hide;
    meta_info_array_[array_pos].fingerprint_[internal_pos] = fp;
    //meta_info_array_[array_pos].bitmap_ |= 1U << internal_pos;
    meta_info_array_[array_pos].set_bitmap(internal_pos);
  }

  inline void set_fingerprint(MetaInfo* new_meta_info_array, unsigned char fp, int pos){
    int array_pos = pos / TABLE_FACTOR;
    int internal_pos = pos & table_factor_hide;
    new_meta_info_array[array_pos].fingerprint_[internal_pos] = fp;
    //new_meta_info_array[array_pos].bitmap_ |= 1U << internal_pos;            
    new_meta_info_array[array_pos].set_bitmap(internal_pos);           
  }

  inline void unset_fingerprint(int pos){
    int array_pos = pos / TABLE_FACTOR;
    int size_limit = real_data_capacity_ / TABLE_FACTOR + 1;
    if(array_pos >= size_limit){
      // Just for debugging
        std::cout << "Wrong insertion in unset" << std::endl;
        exit(-1);
    }
    int internal_pos = pos & table_factor_hide;
    //meta_info_array_[array_pos].bitmap_ &= ~(1U << internal_pos);
    meta_info_array_[array_pos].unset_bitmap(internal_pos);
  }

  int find_free_pos_in_stash(int predicted_pos){
    // according to the predict pos to find the free slot
    int stash_pos = static_cast<int>((stash_capacity_ - 1) * (predicted_pos / static_cast<double>(data_capacity_ - 1)));
    stash_pos = std::max<int>(std::min<int>(stash_pos, stash_capacity_ - 1), 0);
    int curBitmapIdx = stash_pos >> 6;
    uint64_t curBitmapData;

    // Search from curBitmapIdx to (bitmap_size_ - 1)
    for(int i = curBitmapIdx; i < bitmap_size_; ++i){
      curBitmapData = bitmap_[i];
      if(curBitmapData != UINT64_MAX){
        uint64_t bit = extract_rightmost_zero(curBitmapData);
        uint64_t new_value = curBitmapData | bit;
        bitmap_[i] = new_value;
        int real_insert = get_offset(i, bit);
        return real_insert;
      }
    }

    // Search from 0 to (curBitmapIdx - 1)
    for(int i = 0; i < curBitmapIdx; ++i){
      curBitmapData = bitmap_[i];
      if(curBitmapData != UINT64_MAX){
        uint64_t bit = extract_rightmost_zero(curBitmapData);
        uint64_t new_value = curBitmapData | bit;
        bitmap_[i] = new_value;
        int real_insert = get_offset(i, bit);
        return real_insert;
      }
    }

    return -1;
  }

  int find_free_pos_in_stash_with_sync(int predicted_pos){
    // according to the predict pos to find the free slot
    int stash_pos = static_cast<int>((stash_capacity_ - 1) * (predicted_pos / static_cast<double>(data_capacity_ - 1)));
    stash_pos = std::max<int>(std::min<int>(stash_pos, stash_capacity_ - 1), 0);
    int curBitmapIdx = stash_pos >> 6;
    uint64_t curBitmapData;

    // Search from curBitmapIdx to (bitmap_size_ - 1)
    for(int i = curBitmapIdx; i < bitmap_size_; ++i){
      curBitmapData = bitmap_[i];
      while(curBitmapData != UINT64_MAX){
        uint64_t bit = extract_rightmost_zero(curBitmapData);
        uint64_t new_value = curBitmapData | bit;
        if(CAS(&(bitmap_[i]), &curBitmapData, new_value)){
          int real_insert = get_offset(i, bit);
          return real_insert;
        }
      }
    }

    // Search from 0 to (curBitmapIdx - 1)
    for(int i = 0; i < curBitmapIdx; ++i){
      curBitmapData = bitmap_[i];
      while(curBitmapData != UINT64_MAX){
        uint64_t bit = extract_rightmost_zero(curBitmapData);
        uint64_t new_value = curBitmapData | bit;
        if(CAS(&(bitmap_[i]), &curBitmapData, new_value)){
          int real_insert = get_offset(i, bit);
          return real_insert;
        }
      }
    }

    return -1;
  }

  void set_pos_in_stash(int stash_pos){
    int bitmap_pos = stash_pos >> 6;
    int bit_pos = stash_pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Unset a bit in the bitmap of overflow stash
  inline void unset_pos_in_stash_with_sync(int stash_pos){
    if(stash_pos < 0 || stash_pos > stash_capacity_){
      LOG_FATAL("Invalid stash pos during unset");
    }

    int curBitmapIdx = stash_pos >> 6;
    int bit_pos = stash_pos - (curBitmapIdx << 6);
    uint64_t curBitmapData = bitmap_[curBitmapIdx];
    uint64_t newBitmapData = curBitmapData & (~(1ULL << bit_pos));
    while(!CAS(&(bitmap_[curBitmapIdx]), &curBitmapData, newBitmapData)){
      newBitmapData = curBitmapData & (~(1ULL << bit_pos));
    }
  }

  void insert_to_the_node(const T& key, const P& payload){
    unsigned char key_hash = hashcode1B<T>(key);
    int predicted_pos = predict_position(key);
    int insertion_position = find_free_pos_in_range(predicted_pos, predicted_pos + PROBING_LENGTH);
    int parameter_pos = predicted_pos / scale_factor_;

    if(insertion_position < 0){
      // resolve the collision in overflow stash
      insert_to_stash_without_cost(predicted_pos, key, payload, key_hash);
      scale_parameters_[parameter_pos].stash_insert_++;
    }else{
      insert_element_at_without_persistence(key, payload, insertion_position);
      // Need to udpate the metadata array in DRAM
      // To indicate that some elements has been inserted during insertion
      set_fingerprint(key_hash, insertion_position);
      scale_parameters_[parameter_pos].array_insert_++;
    }
    scale_parameters_[parameter_pos].num_keys_++;
  }

  // get the locks in the whole node, and also set that this node is in a SMO
  void get_all_locks_in_node(){
    for(int i = 0; i < scale_parameters_size_; ++i){
      get_lock(i);
    }  
  }

  void release_all_locks_in_node(){
    for(int i = 0; i < scale_parameters_size_; ++i){
      release_lock(i);
    } 
  }

  int get_position_in_PA(const T& key){ 
    return predict_position(key); 
  }

  T& get_key_from_sorted_array(int pos){
    if(sorted_slots_ == nullptr){
      std::cout << "Wrong get key in sorted array!!!" << std::endl;
    }
    return sorted_slots_[pos].first;
  }

  // First value in returned pair is fail flag:
  // 0 if successful insert (possibly with automatic expansion).
  // 1 if no insert because of significant cost deviation.
  // 2 if no insert because of "catastrophic" cost.
  // 3 if no insert because node is at max capacity.
  // 4 if no insert because node is in SMO, the main operation need retry
  // -1 if key already exists and duplicates not allowed.
  //
  // Second value in returned pair is stash_frac, or of the
  // already-existing key.
  // -1 if no insertion.
  // if the operation return because of it needs to do SMO, then return the current version of the SMO lock
  std::pair<int, double> insert(const T& key, const P& payload, OverflowDesc *overflow_desc = nullptr) {
    uint32_t lock_version;
    bool two_lock = false;
    if (test_lock_set(lock_version))
    {
      // If the node is in a SMO state, just return
      return {4, -1};
    }

    int predicted_pos = predict_position(key); 
    int meta_pos = predicted_pos / TABLE_FACTOR;
    int parameter_pos = (meta_pos * TABLE_FACTOR) / scale_factor_;
    // Prefetch the metadata
    __builtin_prefetch(meta_info_array_ + meta_pos, 1);

    unsigned char key_hash = hashcode1B<T>(key);
    // Get the lock of the corresponding partition
    // The prefetching may not take effect since the prefetch distance is too short
    while(!try_get_lock(parameter_pos)){
      if(test_lock_set(lock_version)){ // test whether the node is in a SMO
        return {4, -1};   
      }
    }

    if(((predicted_pos + PROBING_LENGTH) / scale_factor_) != parameter_pos){
      two_lock = true;
      get_lock(parameter_pos + 1);
    }
  
    // duplicate check
    uint32_t compose_bitmap;
    if(linear_probing_negative_search(predicted_pos, key, key_hash, compose_bitmap)){
      if(two_lock){
        release_lock(parameter_pos + 1);
      }
      release_lock(parameter_pos);
      return {-1, -1};
    }

    int insertion_position = find_free_pos_in_range(predicted_pos, predicted_pos + PROBING_LENGTH, compose_bitmap);
    if(insertion_position < 0){
      insert_to_stash_with_sync(predicted_pos, parameter_pos, meta_pos, key, payload, key_hash, overflow_desc);
      scale_parameters_[parameter_pos].stash_insert_++;
    }else{
      scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * (insertion_position - predicted_pos) / 64.0;
      insert_element_at(key, payload, insertion_position);
      // Need to udpate the metadata array in DRAM
      // To indicate that some elements has been inserted during insertion
      set_fingerprint(key_hash, insertion_position);
      scale_parameters_[parameter_pos].array_insert_++;   
    }

    // Update stats
    scale_parameters_[parameter_pos].num_keys_++;
    scale_parameters_[parameter_pos].num_inserts_++;
    
    // These statistics is not critical, no need to persist them
    //if (key > max_key_) {
    if(key_greater(key, max_key_)){
      do{
        auto old_value = load_multiple_type(&max_key_);
        //if(key > old_value){
        if(key_greater(key, old_value)){
          cas_multiple_type(&max_key_, &old_value, key);
        }
      }while(key_greater(key, max_key_));
      num_right_out_of_bounds_inserts_++; // No need for atomicity 
    }

    if (key_less(key, min_key_)) {
      do{
        auto old_value = load_multiple_type(&min_key_);
        if(key_less(key, old_value)){
          cas_multiple_type(&min_key_, &old_value, key);
        }
      }while(key_less(key, min_key_));
      num_left_out_of_bounds_inserts_++; // No need for atomicity 
    }

    if(two_lock){
      release_lock(parameter_pos + 1);
    }

    // Periodically check for catastrophe
    if (scale_parameters_[parameter_pos].num_inserts_ % 64 == 0 && catastrophic_cost()) {

      if(try_get_lock()){
        release_lock(parameter_pos);
        get_all_locks_in_node();      
        int array_insert = 0;
        int stash_insert = 0;
        update_all_cost(&array_insert, &stash_insert);

        double stash_frac = stash_insert / static_cast<double>(num_keys_);

        stash_frac = std::max(0.05, std::min(0.3, stash_frac));
        return {2, stash_frac};
      }
    }

    // Check if node is full (based on expansion_threshold)
    if (scale_parameters_[parameter_pos].num_keys_ >= scale_parameters_[parameter_pos].expansion_threshold_) {    
      int total_keys = total_keys_in_node();       
      if ((total_keys > expansion_threshold_) && try_get_lock())
      {
          release_lock(parameter_pos);
          get_all_locks_in_node();          
          int array_insert = 0;
          int stash_insert = 0;
          update_all_cost(&array_insert, &stash_insert);

          double stash_frac = stash_insert / static_cast<double>(num_keys_);        
          stash_frac = std::max(0.05, std::min(0.3, stash_frac));

          if (catastrophic_cost()) {
            return {2, stash_frac};
          }

          if (significant_cost_deviation()) {       
            return {1, stash_frac};
          } 
        
          if (num_keys_ > max_slots_ * kMinDensity_) {
            return {3, stash_frac};
          }

          // Need resizing 
          return {5, stash_frac};
      }
    }

    release_lock(parameter_pos);
    return {0, 0};
  }

  static void New_from_existing(PMEMoid *ptr, self_type* old_node){
    auto callback = [](PMEMobjpool *pool, void *ptr, void *arg) {
      auto value_ptr = reinterpret_cast<self_type *>(arg);
      auto node_ptr = reinterpret_cast<self_type* >(ptr);
      memcpy(node_ptr, value_ptr, sizeof(self_type));
      node_ptr->first_block_ = nullptr;
      node_ptr->link_lock_ = 0;
      node_ptr->data_slots_ = nullptr;
      node_ptr->P_data_slots_ = OID_NULL; 
      my_alloc::BasePMPool::Persist(&node_ptr->P_data_slots_, sizeof(node_ptr->P_data_slots_));
      my_alloc::BasePMPool::Persist(&node_ptr->first_block_, sizeof(node_ptr->first_block_));
      // No persistence is still correct
      return 0;
    };

    my_alloc::BasePMPool::Allocate(ptr, kCacheLineSize, sizeof(self_type), callback, reinterpret_cast<void*>(old_node));
  }

  // Resize to a larger capacity according to the old node
  // Directly get sorted node from old node
  // And I assume the new node has copied all the content from the old node
  void resize_from_existing(self_type *old_node, double target_density, bool force_retrain = false,
              bool keep_left = false, bool keep_right = false, bool rebuild_sorted_node = false, double stash_frac = 0.05){
    if (rebuild_sorted_node)
    {
      old_node->build_sorted_slots();
    }

    auto old_data_capacity = data_capacity_;
    // Initialization
    initialize(num_keys_, kMinDensity_, stash_frac);
    if (num_keys_ == 0)
    {
      expansion_threshold_ = data_capacity_;
      scale_parameters_[0].expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      return;
    }

    // Retrain model if the number of keys is sufficiently small (under 50)
    if (num_keys_ < 50 || force_retrain) {
      order_iterator_type it(old_node, 0, true);
      LinearModelBuilder<T> builder(&(this->model_));
      for (int i = 0; !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
      }
      builder.build();
      if (keep_left) {
        this->model_.expand(static_cast<double>(old_data_capacity) / num_keys_);
      } else if (keep_right) {
        this->model_.expand(static_cast<double>(old_data_capacity) / num_keys_);
        this->model_.b_ += (data_capacity_ - old_data_capacity);
      } else {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
      }
    } else {
      if (keep_right) {
        this->model_.b_ += (data_capacity_ - old_data_capacity);
      } else if (!keep_left) {
        this->model_.expand(static_cast<double>(data_capacity_) /
                            old_data_capacity);
      }
    }

    // Use simulation method to insert, but needs to first sort it
    int last_position = -1;
    sorted_slots_ = old_node->sorted_slots_;
    for(int i = 0; i < num_keys_; ++i){
      auto key = sorted_slots_[i].first;
      int predicted_position = predict_position(key);
      unsigned char key_hash = hashcode1B<T>(key);
      int actual_position =
        std::max<int>(predicted_position, last_position + 1);
      int parameter_pos = predicted_position / scale_factor_;

      if((actual_position - predicted_position) >= PROBING_LENGTH){
        insert_to_stash_without_cost(predicted_position, key, sorted_slots_[i].second, key_hash);
        scale_parameters_[parameter_pos].stash_insert_++;
      }else{
        last_position = actual_position;
        // insert to main array
        data_slots_[actual_position] = sorted_slots_[i];
        set_fingerprint(key_hash, actual_position);
        scale_parameters_[parameter_pos].array_insert_++;   
      }
      scale_parameters_[parameter_pos].num_keys_++;
    }

    min_key_ = sorted_slots_[0].first;
    max_key_ = sorted_slots_[num_keys_ - 1].first;
    sorted_slots_ = nullptr;

    int new_total_capacity = data_capacity_ + stash_capacity_;
    expansion_threshold_ =
        std::min(std::max(new_total_capacity * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(new_total_capacity));

    int initial_parameter_size = data_capacity_ / scale_factor_; 
    for(int i = 0 ; i < initial_parameter_size; ++i){
      scale_parameters_[i].expansion_threshold_ = static_cast<int>(SCALE_FACTOR * kMaxDensity_);
    }

    if ((data_capacity_ % scale_factor_) != 0)
    {
      scale_parameters_[initial_parameter_size].expansion_threshold_ = std::min(std::max(static_cast<int>((new_total_capacity % SCALE_FACTOR) * kMaxDensity_),
                                             num_keys_ + 1), new_total_capacity);
    }

    contraction_threshold_ = new_total_capacity * kMinDensity_;
    my_alloc::BasePMPool::Persist(data_slots_, sizeof(V) * (new_total_capacity + PROBING_LENGTH - 1));
    my_alloc::BasePMPool::Persist(this, sizeof(self_type));
  }

  inline bool is_append_mostly_right() const {
    return static_cast<double>(num_right_out_of_bounds_inserts_) /
               num_inserts_ >
           kAppendMostlyThreshold;
  }

  inline bool is_append_mostly_left() const {
    return static_cast<double>(num_left_out_of_bounds_inserts_) / num_inserts_ >
           kAppendMostlyThreshold;
  }

  // Insert key into pos. The caller must guarantee that pos is a gap.
  void insert_element_at(const T& key, P payload, int pos) {
    data_slots_[pos].second = payload;
    data_slots_[pos].first = key;
    clwb(&data_slots_[pos]);
    sfence();
  }

  // insert without any cacheline flush
  inline void insert_element_at_without_persistence(const T& key, P payload, int pos) {
    data_slots_[pos] = std::make_pair(key, payload);
  }

  /*** Deletes ***/

  // Erase the left-most key with the input value
  // Returns the number of keys erased (0 or 1)
  int erase_one(const T& key) {
    return erase(key);
  }

  // !!! NO Concurrency support
  // Erase all keys with the input value
  // Returns the number of keys erased (there may be multiple keys with the same
  // value)
  // haven't gotten the lock
  int erase(const T& key) {
    int m = predict_position(key);
    int parameter_pos = m / scale_factor_;
    int end = m + PROBING_LENGTH;
    for(int i = m; i < end; ++i){
      if(key_equal(ALEX_DATA_NODE_KEY_AT(i), key)){ 
        ALEX_DATA_NODE_KEY_AT(i) = invalid_key_;
#ifdef MY_PERSISTENCE
        clwb(&(ALEX_DATA_NODE_KEY_AT(i)));
        sfence();
#endif
        unset_fingerprint(i);
        scale_parameters_[parameter_pos].num_keys_--;
        return 1;
      }
    }
    
    unsigned char fp = hashcode1B<T>(key);
    int meta_pos = m / TABLE_FACTOR;
    OverflowFinger *overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
    while(overflow_finger != nullptr){
      V** overflow_array = overflow_finger->overflow_array_;
      for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
        if((overflow_array[i] != nullptr) && (static_cast<uint8_t>(reinterpret_cast<uint64_t>(overflow_array[i]) >> 56) == fp)){
          V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
          if(key_equal(ret->first, key)){
            ret->first = invalid_key_;
#ifdef MY_PERSISTENCE            
              clwb(&(ret->first));
              sfence();
#endif
              overflow_array[i] = nullptr;
              scale_parameters_[parameter_pos].num_keys_--;
              return 1;
          }
        }
      }
      overflow_finger = overflow_finger->next_;
    }

    return 0; 
  }


  // Erase keys, has concurrency support
  // 0 means operation successfully execute
  // 1 means operation needs retry
  // 2 means operation needs merge the node and the lock of this node has been acquired
  int erase(const T& key, int* num_erased) {
    uint32_t lock_version;
    bool two_lock = false;
    if(test_lock_set(lock_version)){
      return 1;
    }

    int predicted_pos = predict_position(key);
    int meta_pos = predicted_pos / TABLE_FACTOR;
    int parameter_pos = predicted_pos / scale_factor_;

    // Prefetch the metadata
    __builtin_prefetch(meta_info_array_ + meta_pos, 1);
    while(!try_get_lock(parameter_pos)){
      if(test_lock_set(lock_version)){ // test whether the node is in a SMO
        return 1;
      }
    }

    if(((predicted_pos + PROBING_LENGTH) / scale_factor_) != parameter_pos){
      two_lock = true;
      get_lock(parameter_pos + 1);
    }
    *num_erased = 0;
    scale_parameters_[parameter_pos].num_lookups_++;

    bool erase_success = false;
    int end = predicted_pos + PROBING_LENGTH;
    for(int i = predicted_pos; i < end; ++i){
      if(key_equal(ALEX_DATA_NODE_KEY_AT(i), key)){ 
        ALEX_DATA_NODE_KEY_AT(i) = invalid_key_;
#ifdef MY_PERSISTENCE
        clwb(&(ALEX_DATA_NODE_KEY_AT(i)));
        sfence();
#endif
        unset_fingerprint(i);
        scale_parameters_[parameter_pos].num_keys_--;
        scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * (i - predicted_pos) / 64.0;   
        erase_success = true;
        *num_erased = 1;
        break;
      }
    }

    if(!erase_success){
      scale_parameters_[parameter_pos].num_search_cost_ += 4;
      // Need to erase in stash area, in stash array or overflow stash blocks
      unsigned char fp = hashcode1B<T>(key);
      OverflowFinger *overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
      while(overflow_finger != nullptr){
        V** overflow_array = overflow_finger->overflow_array_;
        for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
          if((overflow_array[i] != nullptr) && (static_cast<uint8_t>(reinterpret_cast<uint64_t>(overflow_array[i]) >> 56) == fp)){
            V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
            if(key_equal(ret->first, key)){
              ret->first = invalid_key_;
  #ifdef MY_PERSISTENCE            
              clwb(&(ret->first));
              sfence();
  #endif
              // Delete according to the offset
              uint8_t offset = static_cast<uint8_t>(reinterpret_cast<uint64_t>(overflow_array[i]) >> 48);
              overflow_array[i] = nullptr;
              if (offset & offsetSet)
              {
                // Delete in overflow stash blocks
                int real_offset = static_cast<int>(offset & offsetMask);
                overflow_stash_type *target_stash = reinterpret_cast<overflow_stash_type*>(reinterpret_cast<uint64_t>(ret) - (sizeof(V) * real_offset));
                bool flag = erase_from_overflow_stash(parameter_pos, target_stash, real_offset);
                if(flag == false){
                  LOG_FATAL("ERROR during erase from overflow stash");
                }
              }else{
                int stash_offset = static_cast<int>(ret - stash_slots_);
                // Delete in stash array
                unset_pos_in_stash_with_sync(stash_offset);
              }
              scale_parameters_[parameter_pos].num_keys_--;
              scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * i / 64.0;  
              *num_erased = 1;
              erase_success = true;
              break;
            }
          }
        }
        if(erase_success){
          break;
        }
        overflow_finger = overflow_finger->next_;
        scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * OVERFLOW_FINGER_LENGTH / 64.0;  
      }
    }

    if(two_lock){
      release_lock(parameter_pos + 1);
    }
    release_lock(parameter_pos);
    return 0; 
  }

  // Erase keys with value between start key (inclusive) and end key.
  // Returns the number of keys erased.
  int erase_range(T start_key, T end_key, bool end_key_inclusive = false) {
    int num_erased = 0;
    if(end_key_inclusive){
      for(int i = 0; i < real_data_capacity_; ++i){
        if(key_greaterequal(ALEX_DATA_NODE_KEY_AT(i), start_key) && key_lessequal(ALEX_DATA_NODE_KEY_AT(i), end_key) &&  check_exists(i)){
          int m = predict_position(ALEX_DATA_NODE_KEY_AT(i));
          ALEX_DATA_NODE_KEY_AT(i) = invalid_key_;
          num_erased++;
#ifdef MY_PERSISTENCE
          clwb(&(ALEX_DATA_NODE_KEY_AT(i)));
          sfence();
#endif
          unset_fingerprint(i);
          int parameter_pos = m / scale_factor_;
          scale_parameters_[parameter_pos].num_keys_--;
        }
      }

      //erase in overflow keys
      int meta_size = real_data_capacity_ / TABLE_FACTOR + 1;
      for(int i = 0; i < meta_size; ++i){
        OverflowFinger *overflow_finger = meta_info_array_[i].get_overflow_finger();
        while(overflow_finger != nullptr){
          V** overflow_array = overflow_finger->overflow_array_;
          for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
            if(overflow_array[i] != nullptr){
              V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
              if(key_greaterequal(ret->first, start_key) && key_lessequal(ret->first, end_key)){
                  int m = predict_position(ret->first);
                  ret->first = invalid_key_;
    #ifdef MY_PERSISTENCE            
                  clwb(&(ret->first));
                  sfence();
    #endif
                  overflow_array[i] = nullptr;
                  int parameter_pos = m / scale_factor_;
                  scale_parameters_[parameter_pos].num_keys_--;
                  num_erased++;
              }
            }
          }
          overflow_finger = overflow_finger->next_;
        }
      }
    }else{
      for(int i = 0; i < real_data_capacity_; ++i){
        if(key_greaterequal(ALEX_DATA_NODE_KEY_AT(i), start_key) && key_less(ALEX_DATA_NODE_KEY_AT(i), end_key) &&  check_exists(i)){
          int m = predict_position(ALEX_DATA_NODE_KEY_AT(i));
          num_erased++;
          ALEX_DATA_NODE_KEY_AT(i) = invalid_key_;
#ifdef MY_PERSISTENCE
          clwb(&(ALEX_DATA_NODE_KEY_AT(i)));
          sfence();
#endif
          unset_fingerprint(i);
          int parameter_pos = m / scale_factor_;
          scale_parameters_[parameter_pos].num_keys_--;
        }
      }

      int meta_size = real_data_capacity_ / TABLE_FACTOR + 1;
      for(int i = 0; i < meta_size; ++i){
        OverflowFinger *overflow_finger = meta_info_array_[i].get_overflow_finger();
        while(overflow_finger != nullptr){
          V** overflow_array = overflow_finger->overflow_array_;
          for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
            if(overflow_array[i] != nullptr){
              V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
              if(key_greaterequal(ret->first, start_key) && key_less(ret->first, end_key)){
                  int m = predict_position(ret->first);
                  ret->first = invalid_key_;
    #ifdef MY_PERSISTENCE            
                  clwb(&(ret->first));
                  sfence();
    #endif
                  overflow_array[i] = nullptr;
                  int parameter_pos = m / scale_factor_;
                  scale_parameters_[parameter_pos].num_keys_--;
                  num_erased++;
              }
            }
          }
          overflow_finger = overflow_finger->next_;
        }
      }
    }
    return num_erased;
  }

    // Update the payload, has concurrency support
  // 0 means operation successfully execute
  // 1 means operation needs retry
  int update(const T& key, const P& payload, int* num_updated) {
    uint32_t lock_version;
    bool two_lock = false;
    if(test_lock_set(lock_version)){
      return 1;
    }

    int predicted_pos = predict_position(key);
    int meta_pos = predicted_pos / TABLE_FACTOR;
    int parameter_pos = predicted_pos / scale_factor_;

    // Prefetch the metadata
    __builtin_prefetch(meta_info_array_ + meta_pos, 1);
    while(!try_get_lock(parameter_pos)){
      if(test_lock_set(lock_version)){ // test whether the node is in a SMO
        return 1;
      }
    }

    if(((predicted_pos + PROBING_LENGTH) / scale_factor_) != parameter_pos){
      two_lock = true;
      get_lock(parameter_pos + 1);
    }
    *num_updated = 0;
    scale_parameters_[parameter_pos].num_lookups_++;

    bool update_success = false;
    int end = predicted_pos + PROBING_LENGTH;
    for(int i = predicted_pos; i < end; ++i){
      if(key_equal(ALEX_DATA_NODE_KEY_AT(i), key)){ 
        ALEX_DATA_NODE_PAYLOAD_AT(i) = payload;
#ifdef MY_PERSISTENCE
        clwb(&(ALEX_DATA_NODE_PAYLOAD_AT(i)));
        sfence();
#endif
        scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * (i - predicted_pos) / 64.0;   
        update_success = true;
        *num_updated = 1;
        break;
      }
    }

    if(!update_success){
      scale_parameters_[parameter_pos].num_search_cost_ += 4;
      // Need to erase in stash area, in stash array or overflow stash blocks
      unsigned char fp = hashcode1B<T>(key);
      OverflowFinger *overflow_finger = meta_info_array_[meta_pos].get_overflow_finger();
      while(overflow_finger != nullptr){
        V** overflow_array = overflow_finger->overflow_array_;
        for(int i = 0; i < OVERFLOW_FINGER_LENGTH; ++i){
          if((overflow_array[i] != nullptr) && (static_cast<uint8_t>(reinterpret_cast<uint64_t>(overflow_array[i]) >> 56) == fp)){
            V* ret = reinterpret_cast<V*>(addrMask & reinterpret_cast<uint64_t>(overflow_array[i]));
            if(key_equal(ret->first, key)){
              ret->second = payload;
  #ifdef MY_PERSISTENCE            
              clwb(&(ret->second));
              sfence();
  #endif
              scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * i / 64.0;  
              *num_updated = 1;
              update_success = true;
              break;
            }
          }
        }
        if(update_success){
          break;
        }
        overflow_finger = overflow_finger->next_;
        scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * OVERFLOW_FINGER_LENGTH / 64.0;  
      }
    }

    if(two_lock){
      release_lock(parameter_pos + 1);
    }
    release_lock(parameter_pos);
    return 0; 
  }

  /*** Recovery ***/

  // Recompute the stash fraction
  double compute_stash_frac(){
    int total_stash_insert = 0;
    int total_num_keys = 0;
    for(int i = 0; i < scale_parameters_size_; ++i){
      total_stash_insert += scale_parameters_[i].stash_insert_;
      total_num_keys += scale_parameters_[i].num_keys_;
    }
    return total_stash_insert / static_cast<double>(total_num_keys);
  }

  // Attach the overflow stash to index_overflow_stash
  inline void reattach_overflow_stash(overflow_stash_type *new_overflow_stash){
    int parameter_pos = new_overflow_stash->parameter_pos_;
    index_overflow_stash_type *index_overflow_stash = scale_parameters_[parameter_pos].index_overflow_stash_;

    while(index_overflow_stash){
      for(int i = 0; i < 5; ++i){
        if(!(index_overflow_stash->bitmap_[i] & bitmapSet)){
          scale_parameters_[parameter_pos].last_stash_ = new_overflow_stash;
          scale_parameters_[parameter_pos].overflow_stash_count_++;
          index_overflow_stash->overflow_stash_array_[i] = new_overflow_stash;
          index_overflow_stash->bitmap_[i] = new_overflow_stash->get_bitmap(invalid_key_) | (static_cast<uint16_t>(1) << 15); 
          return;
        }
      }
      index_overflow_stash = index_overflow_stash->next_;
    }

    // Create a new index overflow
    index_overflow_stash_type *new_index;
    align_zalloc((void**)(&new_index), sizeof(index_overflow_stash_type));
    scale_parameters_[parameter_pos].last_stash_ = new_overflow_stash;
    scale_parameters_[parameter_pos].overflow_stash_count_++;
    new_index->overflow_stash_array_[0] = new_overflow_stash;
    new_index->bitmap_[0] = new_overflow_stash->get_bitmap(invalid_key_) | (static_cast<uint16_t>(1) << 15);
  
    // Attach overflow stash to node
    new_index->next_ = scale_parameters_[parameter_pos].index_overflow_stash_;
    scale_parameters_[parameter_pos].index_overflow_stash_ = new_index;
  }

  void recover_node(uint32_t global_version){ 
    //std::cout << "First need to get the lock" << std::endl;
    while (pmemobj_mutex_trylock(my_alloc::BasePMPool::pm_pool_, &recover_lock_) != 0) {
      if(local_version_ == global_version) return;
    }

    if(local_version_ == global_version){
      pmemobj_mutex_unlock(my_alloc::BasePMPool::pm_pool_, &recover_lock_);
      return;
    }

    // Real Recovery Work
    // 0. clear locks
    lock_ = 0;
    link_lock_ = 0;

    //std::cout << "1. Now creating DRAM data structures" << std::endl;
    // 1. Create DRAM data structure
    align_zalloc((void**)&bitmap_, sizeof(uint64_t) * bitmap_size_);
    int slot_bits = stash_capacity_ % 64;
    if (slot_bits != 0)
    {
      bitmap_[bitmap_size_ - 1] = ~((1ULL << slot_bits) - 1);
    }

    align_zalloc((void**)&scale_parameters_, scale_parameters_size_ * sizeof(ScaleParameter));
    int metainfo_num = real_data_capacity_ / TABLE_FACTOR + 1;
    align_zalloc((void**)&meta_info_array_, sizeof(MetaInfo) * (metainfo_num));

    // 2. Remapping the KV records in data array to meta_info_array
    // No need to write PM, just write to DRAM
    // But also needs to collect cost statistics,a dn udpate to the scale_parameters_
    for(int i = 0; i < real_data_capacity_; ++i){
      if(data_slots_[i].first != invalid_key_){
        T& key = data_slots_[i].first;
        // This is the real key, udpate the hash table and fingerprint
        unsigned char key_hash = hashcode1B<T>(key);
        int predicted_pos = predict_position(key);
        int parameter_pos = predicted_pos / scale_factor_;
        scale_parameters_[parameter_pos].num_search_cost_ += sizeof(V) * (i - predicted_pos) / 64.0;
        set_fingerprint(key_hash, i); // Update fingerprint and bitmap
        scale_parameters_[parameter_pos].array_insert_++;
        scale_parameters_[parameter_pos].num_keys_++;
        scale_parameters_[parameter_pos].num_inserts_++;
        //if(key > max_key_) max_key_ = key;
        //if(key < min_key_) min_key_ = key;
        if(key_greater(key, max_key_)) max_key_ = key;
        if(key_less(key, min_key_)) min_key_ = key;
      }
    }

    // 3. Remapping the KV records in statsh area to meta_info_array
    for (int i = 0; i < stash_capacity_; ++i)
    {
      if(stash_slots_[i].first != invalid_key_){
       // printf("%d: The stash key is %.10f, invalid key = %.10f\n", i, stash_slots_[i].first, invalid_key_);
        T& key = stash_slots_[i].first;
        unsigned char key_hash = hashcode1B<T>(key);
        int predicted_pos = predict_position(key);
        int parameter_pos = predicted_pos / scale_factor_;
        // First set the stash bitmap
        set_pos_in_stash(i);
        uint8_t offset = 0;
        int meta_pos = predicted_pos / TABLE_FACTOR;
        insert_overflowfp_to_meta(meta_pos, parameter_pos, stash_slots_ + i, key_hash, offset);
        scale_parameters_[parameter_pos].stash_insert_++;
        scale_parameters_[parameter_pos].num_keys_++;
        scale_parameters_[parameter_pos].num_inserts_++;
        //if(key > max_key_) max_key_ = key;
        if(key_greater(key, max_key_)) max_key_ = key;
        if(key_less(key, min_key_)) min_key_ = key;
        //if(key < min_key_) min_key_ = key;
      }
    }

    //std::cout << "4. start mapping records from overflow stash to linked list" << std::endl;
    // 4. Remapping the KV records in overflow stash area
    overflow_stash_type* overflow_block = first_block_;
    while(overflow_block != nullptr){
      // First attach to scale_parameters
      reattach_overflow_stash(overflow_block);
      // Attach the individual KV records to Overflow Finger
      for(int i = 0; i < 11; ++i){
        if(overflow_block->overflow_slots_[i].first != invalid_key_){
          // Insert to overflow Finger
          T& key = overflow_block->overflow_slots_[i].first;
          unsigned char key_hash = hashcode1B<T>(key);
          int predicted_pos = predict_position(key);
          int parameter_pos = predicted_pos / scale_factor_;
          uint8_t offset = offsetSet | static_cast<uint8_t>(i);
          int meta_pos = predicted_pos / TABLE_FACTOR;
          insert_overflowfp_to_meta(meta_pos, parameter_pos, overflow_block->overflow_slots_ + i, key_hash, offset);
          scale_parameters_[parameter_pos].stash_insert_++;
          scale_parameters_[parameter_pos].num_keys_++;
          scale_parameters_[parameter_pos].num_inserts_++;
          //if(key > max_key_) max_key_ = key;
          //if(key < min_key_) min_key_ = key;
          if(key_greater(key, max_key_)) max_key_ = key;
          if(key_less(key, min_key_)) min_key_ = key;
        }
      }

      overflow_stash_type* next_overflow_block = overflow_block;
      while(!OID_IS_NULL(next_overflow_block->next_)){
        next_overflow_block = reinterpret_cast<overflow_stash_type*>(pmemobj_direct(next_overflow_block->next_));
        reattach_overflow_stash(next_overflow_block);
        for(int i = 0; i < 11; ++i){
          if(next_overflow_block->overflow_slots_[i].first != invalid_key_){
            // Insert to overflow Finger
            T& key = next_overflow_block->overflow_slots_[i].first;
            unsigned char key_hash = hashcode1B<T>(key);
            int predicted_pos = predict_position(key);
            int parameter_pos = predicted_pos / scale_factor_;
            uint8_t offset = offsetSet | static_cast<uint8_t>(i);
            int meta_pos = predicted_pos / TABLE_FACTOR;
            insert_overflowfp_to_meta(meta_pos, parameter_pos, next_overflow_block->overflow_slots_ + i, key_hash, offset);
            scale_parameters_[parameter_pos].stash_insert_++;
            scale_parameters_[parameter_pos].num_keys_++;
            scale_parameters_[parameter_pos].num_inserts_++;
            //if(key > max_key_) max_key_ = key;
            //if(key < min_key_) min_key_ = key;
            if(key_greater(key, max_key_)) max_key_ = key;
            if(key_less(key, min_key_)) min_key_ = key;
          }
        }
      }

      overflow_block = overflow_block->link_;
    }  
    local_version_ = global_version;
    my_alloc::BasePMPool::Persist(this, sizeof(self_type));
    pmemobj_mutex_unlock(my_alloc::BasePMPool::pm_pool_, &recover_lock_);
  }

  /*** Stats ***/

  // Total size of node metadata
  long long node_size() const override { return sizeof(self_type); }

  // Total size in bytes of key/payload/data_slots and bitmap
  long long data_size() const {
    long long data_size = (data_capacity_ + stash_capacity_) * sizeof(T);
    data_size += (data_capacity_ + stash_capacity_) * sizeof(P);
    data_size += bitmap_size_ * sizeof(uint64_t);
    return data_size;
  }

  /*** Analysis ***/

  uint64_t get_PM_size(){
    uint64_t pm_size = sizeof(self_type);
    pm_size += sizeof(V) * (real_data_capacity_ + stash_capacity_);
    for(int i = 0; i < scale_parameters_size_; ++i){
      index_overflow_stash_type *index_overflow_stash = scale_parameters_[i].index_overflow_stash_;
      while(index_overflow_stash){
        for(int i = 0; i < 5; ++i){
          auto bitmap = index_overflow_stash->bitmap_[i];
          if(bitmap & bitmapSet){
            pm_size += sizeof(overflow_stash_type);
          }
        }
        index_overflow_stash = index_overflow_stash->next_;
      }
    }
    return pm_size;
  }

  uint64_t get_static_slots(){ 
    return (real_data_capacity_ + stash_capacity_);
  }

  uint64_t get_DRAM_size(){
    uint64_t dram_size = 0;
    dram_size += sizeof(uint64_t) * bitmap_size_;
    dram_size += scale_parameters_size_ * sizeof(ScaleParameter);
    int metainfo_num = real_data_capacity_ / TABLE_FACTOR + 1;
    dram_size += metainfo_num * sizeof(MetaInfo);

    // index overflow stash1
    for(int i = 0; i < scale_parameters_size_; ++i){
      index_overflow_stash_type *index_overflow_stash = scale_parameters_[i].index_overflow_stash_;
      while(index_overflow_stash){
        dram_size += sizeof(index_overflow_stash_type);
        index_overflow_stash = index_overflow_stash->next_;
      }
    }

    // overflow finger
    for(int i = 0; i < metainfo_num; ++i){
      auto cur_overflow_finger = meta_info_array_[i].get_overflow_finger();
      while(cur_overflow_finger){
        dram_size += sizeof(OverflowFinger);
        cur_overflow_finger = cur_overflow_finger->next_;
      }
    }
    return dram_size;
  }

  uint64_t get_node_size(){ 
    //uint64_t pm_size = sizeof(self_type);
    uint64_t pm_size = sizeof(V) * (real_data_capacity_ + stash_capacity_);
    return pm_size;
  }

  /*** Debugging ***/

  std::string to_string() const {
    std::string str;
    str += "Num keys: " + std::to_string(num_keys_) + ", Capacity: " +
           std::to_string(data_capacity_) + ", Expansion Threshold: " +
           std::to_string(expansion_threshold_) + "\n";
    for (int i = 0; i < data_capacity_; i++) {
      str += (std::to_string(ALEX_DATA_NODE_KEY_AT(i)) + " ");
    }
    return str;
  }
};

}
