// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/* This file contains the classes for linear models and model builders, helpers
 * for the bitmap,
 * cost model weights, statistic accumulators for collecting cost model
 * statistics,
 * and other miscellaneous functions
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <bitset>
#include <cassert>
#ifdef _WIN32
#include <intrin.h>
#include <limits.h>
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif

#ifdef _MSC_VER
#define forceinline __forceinline
#elif defined(__GNUC__)
#define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define forceinline inline __attribute__((__always_inline__))
#else
#define forceinline inline
#endif
#else
#define forceinline inline
#endif


#define bitScan(x)  __builtin_ffs(x)

#define SIMD_CMP8(src, key)                                         \
  do {                                                              \
    const __m256i key_data = _mm256_set1_epi8(key);                 \
    __m256i seg_data =                                              \
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)); \
    __m256i rv_mask = _mm256_cmpeq_epi8(seg_data, key_data);        \
    mask = _mm256_movemask_epi8(rv_mask);                           \
  } while (0)

#define SSE_CMP8(src, key)                                       \
  do {                                                           \
    const __m128i key_data = _mm_set1_epi8(key);                 \
    __m128i seg_data =                                           \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
    __m128i rv_mask = _mm_cmpeq_epi8(seg_data, key_data);        \
    mask = _mm_movemask_epi8(rv_mask);                           \
  } while (0)


namespace alex {
  
// global statistics about #insert/search iterations
uint64_t insert_iterations = 0;
uint64_t insert_times = 0;
uint64_t search_iterations = 0;
uint64_t search_times = 0;
uint64_t max_search_iterations = 0;
uint64_t max_insert_iterations = 0;
uint64_t wrong_positive_search_times = 0;
uint64_t positive_search_times = 0;
uint64_t overflow_search_times = 0;

// SMO statistics
uint64_t resize_without_decision = 0;
uint64_t resize_with_decision = 0;
uint64_t split_times = 0;
uint64_t DN_split_side = 0;
uint64_t DN_split_downward = 0;
uint64_t IN_expand = 0;
uint64_t cata_cost_times = 0;
uint64_t deviation_times = 0;

uint64_t model_traversal_ms = 0;
uint64_t bitmap_lookup_ms = 0;
uint64_t data_lookup_ms = 0;
uint64_t overflow_lookup_ms = 0;
uint64_t traversal_levels = 0;
uint64_t data_lookup_nums = 0;
uint64_t overflow_traversal_nums = 0;
uint64_t overflow_traversal_level_nums = 0;
uint64_t meta_array_traversal_nums = 0; 

const uint32_t lockSet = ((uint32_t)1 << 31);
const uint32_t lockMask = ((uint32_t)1 << 31) - 1;
const int counterMask = (1 << 19) - 1;

const uint16_t bitmapSet = ((uint16_t)1 << 15);
const uint8_t offsetSet = ((uint8_t)1 << 7);
const uint8_t offsetMask = offsetSet - 1;
const uint64_t fullOffsetMask = ((uint64_t)1 << 8) - 1;
const uint16_t bitmapMask = ((((uint16_t)1 << 4) - 1) << 11);
const uint64_t addrMask = ((1ULL << 48) - 1);
const uint64_t headerMask = (((1ULL << 16) - 1) << 48);

#define OVERFLOW_FINGER_LENGTH 15
#define PROBING_LENGTH 16
#define TABLE_FACTOR 16
#define SCALE_FACTOR 256
#define LOG_SMO 1 // Use log to ensure the crash consistency of SMO
#define FINGERPRINT 1
#define DUPLICATE_CHECK 1
#define NEW_COST_MODEL 1
#define CONCURRENCY 1
#define NEW_LOCK 1 


int parameter_mask = ~((1 << 4) - 1);
uint64_t array_num_write = 0;
uint64_t link_num_write = 0;
// during normal inserts
uint64_t link_block_alloc = 0;
// during SMO
uint64_t smo_link_block_alloc = 0;
uint64_t smo_num_write = 0;
uint64_t insert_num_keys = 0;
uint64_t keys_in_main = 0;
uint64_t keys_in_overflow = 0;
uint64_t main_array_slots = 0;
uint64_t all_slots = 0;
uint64_t main_array_and_block = 0;
double max_overflow_ratio = 0;
double min_overflow_ratio = 1;
double total_overflow_frac = 0;
double total_overflow_frac_with_weight = 0;
double total_overflow_frac_after_adjust = 0;
double total_overflow_frac_with_weight_after_adjust = 0;
double positive_predict_overflow = 0;
double positive_predict_overflow_with_weight = 0;
double positive_predict_overflow_keys = 0;
double positive_predict_overflow_times = 0;
double negative_predict_overflow = 0;
double negative_predict_overflow_with_weight = 0;
double negative_predict_overflow_keys = 0;
double negative_predict_overflow_times = 0;
double max_predict_overflow_ratio = 0;
double min_predict_overflow_ratio = 1;
uint64_t generated_data_nodes = 0;
uint64_t smo_times = 0;
uint64_t cata_times = 0;
uint64_t smo_rewrite_keys = 0;
double total_set_overflow_frac = 0;
uint64_t stash_insert = 0;
uint64_t overflow_stash_insert = 0;
double total_stash_frac = 0;


constexpr double log_length_ = std::log2(PROBING_LENGTH);
constexpr int table_factor_hide = (1 << 4) - 1;

/*** hash function ***/
template <class T>
static inline unsigned char hashcode1B(T y) {
    uint64_t x = (uint64_t &)y;
    x ^= x>>32;
    x ^= x>>16;
    x ^= x>>8;
    return (unsigned char)(x&0x0ffULL);
}

void align_alloc(void **ptr, size_t size){
  posix_memalign(ptr, 64, size);
}

void align_zalloc(void **ptr, size_t size){
  posix_memalign(ptr, 64, size);
  memset(*ptr, 0, size);
}

/*** Linear model and model builder ***/

// Forward declaration
template <class T>
class LinearModelBuilder;

// Linear regression model
template <class T>
class LinearModel {
 public:
  double a_ = 0;  // slope
  double b_ = 0;  // intercept

  LinearModel() = default;
  LinearModel(double a, double b) : a_(a), b_(b) {}
  explicit LinearModel(const LinearModel& other) : a_(other.a_), b_(other.b_) {}

  void expand(double expansion_factor) {
    a_ *= expansion_factor;
    b_ *= expansion_factor;
  }

  inline int predict(T key) const {
    return static_cast<int>(a_ * static_cast<double>(key) + b_);
  }

  inline double predict_double(T key) const {
    return a_ * static_cast<double>(key) + b_;
  }
};

template <class T>
class LinearModelBuilder {
 public:
  LinearModel<T>* model_;

  explicit LinearModelBuilder<T>(LinearModel<T>* model) : model_(model) {}

  inline void add(T x, int y) {
    count_++;
    x_sum_ += static_cast<long double>(x);
    y_sum_ += static_cast<long double>(y);
    xx_sum_ += static_cast<long double>(x) * x;
    xy_sum_ += static_cast<long double>(x) * y;
    x_min_ = std::min<T>(x, x_min_);
    x_max_ = std::max<T>(x, x_max_);
    y_min_ = std::min<double>(y, y_min_);
    y_max_ = std::max<double>(y, y_max_);
  }

  void build() {
    if (count_ <= 1) {
      model_->a_ = 0;
      model_->b_ = static_cast<double>(y_sum_);
      return;
    }

    if (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_ == 0) {
      // all values in a bucket have the same key.
      model_->a_ = 0;
      model_->b_ = static_cast<double>(y_sum_) / count_;
      return;
    }

    auto slope = static_cast<double>(
        (static_cast<long double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
        (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_));
    auto intercept = static_cast<double>(
        (y_sum_ - static_cast<long double>(slope) * x_sum_) / count_);
    model_->a_ = slope;
    model_->b_ = intercept;

    // If floating point precision errors, fit spline
    if (model_->a_ <= 0) {
      model_->a_ = (y_max_ - y_min_) / (x_max_ - x_min_);
      model_->b_ = -static_cast<double>(x_min_) * model_->a_;
    }
  }

 private:
  int count_ = 0;
  long double x_sum_ = 0;
  long double y_sum_ = 0;
  long double xx_sum_ = 0;
  long double xy_sum_ = 0;
  T x_min_ = std::numeric_limits<T>::max();
  T x_max_ = std::numeric_limits<T>::lowest();
  double y_min_ = std::numeric_limits<double>::max();
  double y_max_ = std::numeric_limits<double>::lowest();
};

/*** Comparison ***/

struct AlexCompare {
  template <class T1, class T2>
  bool operator()(const T1& x, const T2& y) const {
    static_assert(
        std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
        "Comparison types must be numeric.");
    return x < y;
  }
};

/*** Helper methods for bitmap ***/

// Extract the rightmost 1 in the binary representation.
// e.g. extract_rightmost_one(010100100) = 000000100
inline uint64_t extract_rightmost_one(uint64_t value) {
  return value & -static_cast<int64_t>(value);
}

inline uint32_t extract_rightmost_one(uint32_t value) {
  return value & -static_cast<int32_t>(value);
}

// Extract the rightmost 0 in the binary representation
inline uint64_t extract_rightmost_zero(uint64_t value) {
  uint64_t rvalue = ~value;
  return rvalue & -static_cast<int64_t>(rvalue);
}

// Extract the rightmost 0 in the binary representation
inline uint32_t extract_rightmost_zero(uint32_t value) {
  uint32_t rvalue = ~value;
  return rvalue & -static_cast<int32_t>(rvalue);
}

// Remove the rightmost 1 in the binary representation.
// e.g. remove_rightmost_one(010100100) = 010100000
inline uint64_t remove_rightmost_one(uint64_t value) {
  return value & (value - 1);
}

inline uint32_t remove_rightmost_one(uint32_t value) {
  return value & (value - 1);
}

// Count the number of 1s in the binary representation.
// e.g. count_ones(010100100) = 3
inline int count_ones(uint64_t value) {
  return static_cast<int>(_mm_popcnt_u64(value));
}

inline int count_ones(uint32_t value) {
  return static_cast<int>(_mm_popcnt_u32(value));
}


// Get the offset of a bit in a bitmap.
// word_id is the word id of the bit in a bitmap
// bit is the word that contains the bit
inline int get_offset(int word_id, uint64_t bit) {
  return (word_id << 6) + count_ones(bit - 1);
}

/*** Cost model weights ***/

// Intra-node cost weights
double kExpSearchIterationsWeight = 20;
double kShiftsWeight = 0.5;

// New intra-node cost weights
double kSearchCostWeight = 20;
double kInsertCostWeight = 20;

// TraverseToLeaf cost weights
double kNodeLookupsWeight = 20;
double kModelSizeWeight = 5e-7;

/*** Stat Accumulators ***/

struct DataNodeStats {
#ifdef NEW_COST_MODEL
  double num_search_cost = 0;
  double num_insert_cost = 0;
  double overflow_frac = 0;
#else
  double num_search_iterations = 0;
  double num_shifts = 0;
#endif
};

// Used when stats are computed using a sample
struct SampleDataNodeStats {
  double log2_sample_size = 0;
  double num_search_iterations = 0;
  double log2_num_shifts = 0;
};

// Accumulates stats that are used in the cost model, based on the actual vs
// predicted position of a key
class StatAccumulator {
 public:
  virtual ~StatAccumulator() = default;
  virtual void accumulate(int actual_position, int predicted_position) = 0;
#ifdef NEW_COST_MODEL  
  virtual void overflow_accumulate(int overflow_num) = 0;
#endif
  virtual double get_stat() = 0;
  virtual void reset() = 0;
};

// Mean log error represents the expected number of exponential search
// iterations when doing a lookup
class ExpectedSearchIterationsAccumulator : public StatAccumulator {
 public:
  void accumulate(int actual_position, int predicted_position) override {
    cumulative_log_error_ +=
        std::log2(std::abs(predicted_position - actual_position) + 1);
    count_++;
  }

#ifdef NEW_COST_MODEL
  void overflow_accumulate(int overflow_num) override {

  }
#endif

  double get_stat() override {
    if (count_ == 0) return 0;
    return cumulative_log_error_ / count_;
  }

  void reset() override {
    cumulative_log_error_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_log_error_ = 0;
  int count_ = 0;
};

// Mean shifts represents the expected number of shifts when doing an insert
class ExpectedShiftsAccumulator : public StatAccumulator {
 public:
  explicit ExpectedShiftsAccumulator(int data_capacity)
      : data_capacity_(data_capacity) {}

  // A dense region of n keys will contribute a total number of expected shifts
  // of approximately
  // ((n-1)/2)((n-1)/2 + 1) = n^2/4 - 1/4
  // This is exact for odd n and off by 0.25 for even n.
  // Therefore, we track n^2/4.
  void accumulate(int actual_position, int) override {
    if (actual_position > last_position_ + 1) {
      long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
      num_expected_shifts_ += (dense_region_length * dense_region_length) / 4;
      dense_region_start_idx_ = actual_position;
    }
    last_position_ = actual_position;
    count_++;
  }

#ifdef NEW_COST_MODEL
  void overflow_accumulate(int overflow_num) override {
    
  }
#endif

  double get_stat() override {
    if (count_ == 0) return 0;
    // first need to accumulate statistics for current packed region
    long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
    long long cur_num_expected_shifts =
        num_expected_shifts_ + (dense_region_length * dense_region_length) / 4;
    return cur_num_expected_shifts / static_cast<double>(count_);
  }

  void reset() override {
    last_position_ = -1;
    dense_region_start_idx_ = 0;
    num_expected_shifts_ = 0;
    count_ = 0;
  }

 public:
  int last_position_ = -1;
  int dense_region_start_idx_ = 0;
  long long num_expected_shifts_ = 0;
  int count_ = 0;
  int data_capacity_ = -1;  // capacity of node
};

// Combines ExpectedSearchIterationsAccumulator and ExpectedShiftsAccumulator
class ExpectedIterationsAndShiftsAccumulator : public StatAccumulator {
 public:
  ExpectedIterationsAndShiftsAccumulator() = default;
  explicit ExpectedIterationsAndShiftsAccumulator(int data_capacity)
      : data_capacity_(data_capacity) {}

  void accumulate(int actual_position, int predicted_position) override {
    cumulative_log_error_ +=
        std::log2(std::abs(predicted_position - actual_position) + 1);

    if (actual_position > last_position_ + 1) {
      long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
      num_expected_shifts_ += (dense_region_length * dense_region_length) / 4;
      dense_region_start_idx_ = actual_position;
    }
    last_position_ = actual_position;

    count_++;
  }

#ifdef NEW_COST_MODEL
  void overflow_accumulate(int overflow_num) override {
    
  }
#endif

  double get_stat() override {
    assert(false);  // this should not be used
    return 0;
  }

  double get_expected_num_search_iterations() {
    if (count_ == 0) return 0;
    return cumulative_log_error_ / count_;
  }

  double get_expected_num_shifts() {
    if (count_ == 0) return 0;
    long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
    long long cur_num_expected_shifts =
        num_expected_shifts_ + (dense_region_length * dense_region_length) / 4;
    return cur_num_expected_shifts / static_cast<double>(count_);
  }

  void reset() override {
    cumulative_log_error_ = 0;
    last_position_ = -1;
    dense_region_start_idx_ = 0;
    num_expected_shifts_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_log_error_ = 0;
  int last_position_ = -1;
  int dense_region_start_idx_ = 0;
  long long num_expected_shifts_ = 0;
  int count_ = 0;
  int data_capacity_ = -1;  // capacity of node
};


// Mean log error represents the expected number of exponential search
// iterations when doing a lookup
class ExpectedSearchCostAccumulator : public StatAccumulator {
   public:
  ExpectedSearchCostAccumulator(size_t cost_unit, int overflow_block_unit) : cost_unit_(cost_unit), overflow_block_unit_(overflow_block_unit){

  }

  ExpectedSearchCostAccumulator() {
    cost_unit_ = 16; // sizeof(V) = 16
    overflow_block_unit_ = 14;
  }

  void accumulate(int actual_position, int predicted_position) override {
    if(actual_position < predicted_position){
      printf("wrong position 1!!!\n");
      exit(-1);
    }
    cumulative_search_cost_ += (actual_position - predicted_position) * cost_unit_ / 64.0;
    count_++;

  }

#ifdef NEW_COST_MODEL
  void overflow_accumulate(int overflow_num) override {
    /*
    cumulative_search_cost_ += (overflow_num * cost_unit_) / 64.0
                              + overflow_num / overflow_block_unit_
                              + (overflow_num % overflow_block_unit_) * cost_unit_ / 64.0 + 1;
    */
    cumulative_search_cost_ +=  ((overflow_num + PROBING_LENGTH) * cost_unit_) / 64.0 + 1;                              
    count_++;
    overflow_count_++;
    /*
    cumulative_search_cost_ +=  (overflow_num * PROBING_LENGTH * cost_unit_) / 64.0 + overflow_num / overflow_block_unit_ + 1;
    count_+=overflow_num;
    */
  }
#endif

  double get_stat() override {
    if (count_ == 0) return 0;
    return cumulative_search_cost_ / static_cast<double>(count_);
  }

  void reset() override {
    cumulative_search_cost_ = 0;
    count_ = 0;
  }

  double get_overflow_frac() {
    return overflow_count_ / static_cast<double>(count_);
  }

 public:
  double cumulative_search_cost_ = 0;
  int count_ = 0;
  int overflow_count_ = 0;
  size_t cost_unit_;
  int overflow_block_unit_;
};

// Combines ExpectedSearchIterationsAccumulator and ExpectedShiftsAccumulator
class ExpectedInsertAndSearchCostAccumulator : public StatAccumulator {
 public:
  ExpectedInsertAndSearchCostAccumulator(){
    cost_unit_ = 16;
    overflow_block_unit_ = 14;
  }
  explicit ExpectedInsertAndSearchCostAccumulator(size_t cost_unit, int overflow_block_unit)
      : cost_unit_(cost_unit), overflow_block_unit_(overflow_block_unit) {}

  void accumulate(int actual_position, int predicted_position) override {
    if(actual_position < predicted_position){
      printf("wrong position 2!!!\n");
      exit(-1);
    }
    cumulative_search_cost_ += (actual_position - predicted_position) * cost_unit_ / 64.0;
    count_++;
  }

#ifdef NEW_COST_MODEL
  void overflow_accumulate(int overflow_num) override {
    auto whether_first = overflow_num % overflow_block_unit_;
    cumulative_search_cost_ += ((overflow_num + PROBING_LENGTH) * cost_unit_) / 64.0;
    
    if(whether_first == 1){      
      cumulative_insert_cost_ += 2.5;
    }else{
      cumulative_insert_cost_ += 0.5;
    }
    
    count_++;
    overflow_count_++;
  }
#endif

  double get_stat() override {
    assert(false);  // this should not be used
    return 0;
  }

  double get_expected_search_cost() {
    if (count_ == 0) return 0;
    return cumulative_search_cost_ / static_cast<double>(count_);
  }

  double get_expected_insert_cost() {
    if (count_ == 0) return 0;
    return cumulative_insert_cost_ / static_cast<double>(count_);
  }

  double get_overflow_frac() {
    return overflow_count_ / static_cast<double>(count_);
  }

  void reset() override {
    cumulative_search_cost_ = 0;
    cumulative_insert_cost_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_search_cost_ = 0;
  double cumulative_insert_cost_ = 0;
  int count_ = 0;
  int overflow_count_ = 0;
  size_t cost_unit_;
  int overflow_block_unit_;
};

/*** Miscellaneous helpers ***/

// https://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
inline int pow_2_round_up(int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

// https://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
inline int log_2_round_down(int x) {
  int res = 0;
  while (x >>= 1) ++res;
  return res;
}

// https://stackoverflow.com/questions/1666093/cpuid-implementations-in-c
class CPUID {
  uint32_t regs[4];

 public:
  explicit CPUID(unsigned i, unsigned j) {
#ifdef _WIN32
    __cpuidex((int*)regs, (int)i, (int)j);
#else
    asm volatile("cpuid"
                 : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                 : "a"(i), "c"(j));
#endif
  }

  const uint32_t& EAX() const { return regs[0]; }
  const uint32_t& EBX() const { return regs[1]; }
  const uint32_t& ECX() const { return regs[2]; }
  const uint32_t& EDX() const { return regs[3]; }
};

// https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
bool cpu_supports_bmi() {
  return static_cast<bool>(CPUID(7, 0).EBX() & (1 << 3));
}
}
