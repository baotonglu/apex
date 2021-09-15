// Define the benchmarking operations
#pragma once

#include "../tree.h"
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sched.h>
#include <chrono>
#include "../util/utils.h"
#include "../util/allocator.h"
#include <cmath>

#define EPOCH_DURATION 1000
#define TO_SCAN 100

/*** Benchmark Control Function ***/

bool finished = false;
int bar_a, bar_b, bar_c;
std::mutex mtx;
std::condition_variable cv;

struct operation_record_t {
  uint64_t number = 0;
  uint64_t dummy[8]; /*patch to a cacheline size, avoid false sharing*/
};

operation_record_t operation_record[1024]; // Used for sampling

void set_affinity(uint32_t idx) {
  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  CPU_SET(idx, &my_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

inline void spin_wait() {
  SUB(&bar_b, 1);
  while (LOAD(&bar_a) == 1)
    ; /*spinning*/
}

inline void end_notify() {
  if (SUB(&bar_c, 1) == 0) {
    std::unique_lock<std::mutex> lck(mtx);
    finished = true;
    cv.notify_one();
  }
}

struct Range {
  int id; // To specify which core to attach
  uint64_t num; // Total insertion num
  char *workload; // Real workload, and please ensure this is the start of 
  double total_time; // Consumed time
};

/*** Benchmark Single Operation ***/

template <class T, class P>
void concurr_insert(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t round = num / EPOCH_DURATION;
    uint64_t i = 0;
    int fail_insert = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    while (i < round) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      uint64_t end = (i + 1) * EPOCH_DURATION;
      for (uint64_t j = i * EPOCH_DURATION; j < end; ++j) {
        bool ret = index->insert(key_array[j], reinterpret_cast<P>(&key_array[j]));
        if(!ret) fail_insert++;
      }
      ++i;
    }

    {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      for (i = EPOCH_DURATION * round; i < num; ++i) {
        bool ret = index->insert(key_array[i], reinterpret_cast<P>(&key_array[i]));
        if(!ret) fail_insert++;
      }
    }

    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Fail insert = " << fail_insert << std::endl;
    end_notify();
}

// Without Epoch operation
template <class T, class P>
void insert_without_epoch(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    int fail_insert = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < num; ++i) {
      bool ret = index->insert(key_array[i], reinterpret_cast<P>(&key_array[i]));
      if(!ret) fail_insert++;
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Fail insert = " << fail_insert << std::endl;
    end_notify();
}

template <class T, class P>
void concurr_search(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t round = num / EPOCH_DURATION;
    uint64_t i = 0;
    uint64_t found = 0;
    P payload;
    uint64_t total_num = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    while (i < round) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      uint64_t end = (i + 1) * EPOCH_DURATION;
      for (uint64_t j = i * EPOCH_DURATION; j < end; ++j) {
         bool ret = index->search(key_array[j], &payload);
         if(ret){
           found++;
           total_num += reinterpret_cast<uint64_t>(payload);
         }
      }
      ++i;
    }

    {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      for (i = EPOCH_DURATION * round; i < num; ++i) {
        bool ret = index->search(key_array[i], &payload);
        if(ret){
          found++;
          total_num += reinterpret_cast<uint64_t>(payload);
        }
      }
    }

    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during search = " << num - found << " with total " << total_num << std::endl;   
    end_notify();
}

// Without Epoch operation
template <class T, class P>
void search_without_epoch(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t found = 0;
    P payload;
    uint64_t total_num = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < num; ++i) {
      bool ret = index->search(key_array[i], &payload);
      if(ret){
        found++;
        total_num += reinterpret_cast<uint64_t>(payload);
      }
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during search = " << num - found << " with total " << total_num << std::endl;   
    end_notify();
}


template <class T, class P>
void concurr_range(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t round = num / EPOCH_DURATION;
    uint64_t i = 0;
    uint64_t found = 0;
    uint64_t not_enough = 0;
    typedef std::pair<T, P> V;
    V* result = new V[TO_SCAN];

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    while (i < round) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      uint64_t end = (i + 1) * EPOCH_DURATION;
      for (uint64_t j = i * EPOCH_DURATION; j < end; ++j) {
        found = index->range_scan_by_size(key_array[j], TO_SCAN, result);
        if(found != TO_SCAN){
          not_enough++;
        }
      }
      ++i;
    }

    {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      for (i = EPOCH_DURATION * round; i < num; ++i) {
        found = index->range_scan_by_size(key_array[i], TO_SCAN, result);
        if(found != TO_SCAN){
          not_enough++;
        }
      }
    }

    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not enough records during range = " << not_enough << std::endl;   
    end_notify();
}

// Without Epoch operation
template <class T, class P>
void range_without_epoch(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t found = 0;
    typedef std::pair<T, P> V;
    V* result = new V[TO_SCAN];
    uint64_t not_enough = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < num; ++i) {
      found = index->range_scan_by_size(key_array[i], TO_SCAN, result);
      if(found != TO_SCAN) {
        not_enough++;
      }
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not enough records during search = " << not_enough << std::endl;   
    end_notify();
}

// Debug the correctness of range query
template <class T, class P>
void range_debug(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload); // This array is also the sorted keys that is indexes by the index
    uint64_t found = 0;
    typedef std::pair<T, P> V;
    V* result = new V[TO_SCAN];
    uint64_t not_enough = 0;
    // Random num generator
    // std::mt19937_64 gen(std::random_device{}());
    std::mt19937_64 gen(111146);
    std::uniform_int_distribution<int> dis(0, num - 1);
    std::cout.precision(10);

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < num; ++i) {
      int pos = dis(gen);
      found = index->range_scan_by_size(key_array[pos], TO_SCAN, result);
      if((num - pos >= TO_SCAN) && (found != TO_SCAN)) {
        not_enough++;
        std::cout << "pos = " << pos << std::endl;
        std::cout << "not enough key = " << key_array[pos] << std::endl;
        std::cout << "not enough scan = " << not_enough << std::endl;
      }

      // Verif the correctness of read record
      for (int j = 0; j < found; ++j)
      {
        if(result[j].first != key_array[pos + j]){
          std::cout << "Stop!!!" << std::endl;
          std::cout << "pos = " << pos << std::endl;
          std::cout << "i = " << i << std::endl;
          std::cout << "j = " << j << std::endl;
          std::cout << "search key = " << key_array[pos] << std::endl;
          std::cout << "result at differ = " << result[j].first << std::endl;
          std::cout << "key array = " << key_array[pos + j] << std::endl;
          std::cout << "--------------------start list all keys in result array----------------------" << std::endl;
          for(int k = 0; k < found; ++k){
            std::cout << "Result " << k << " = " << result[k].first << std::endl;
            std::cout << "Coorect result " << k << " = " << key_array[pos + k] << std::endl;
          }
          exit(-1);
        }
      }
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during search = " << not_enough << std::endl;   
    end_notify();
}


template <class T, class P>
void concurr_erase(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t round = num / EPOCH_DURATION;
    uint64_t i = 0;
    uint64_t erased = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    while (i < round) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      uint64_t end = (i + 1) * EPOCH_DURATION;
      for (uint64_t j = i * EPOCH_DURATION; j < end; ++j) {
        if(index->erase(key_array[j])){
          erased++;
        }
      }
      ++i;
    }

    {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      for (i = EPOCH_DURATION * round; i < num; ++i) {
        if(index->erase(key_array[i])){
          erased++;
        }
      }
    }

    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during erase = " << num - erased << std::endl;
    end_notify();
}

// Without Epoch operation
template <class T, class P>
void erase_without_epoch(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t erased = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < num; ++i) {
      if(index->erase(key_array[i])){
        erased++;
      }
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during erase = " << num - erased << std::endl;
    end_notify();
}

template <class T, class P>
void concurr_update(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t round = num / EPOCH_DURATION;
    uint64_t i = 0;
    uint64_t updates = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    while (i < round) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      uint64_t end = (i + 1) * EPOCH_DURATION;
      for (uint64_t j = i * EPOCH_DURATION; j < end; ++j) {
        if(index->update(key_array[j], reinterpret_cast<P>(&key_array[j]))){
          updates++;
        }
      }
      ++i;
    }

    {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      for (i = EPOCH_DURATION * round; i < num; ++i) {
        if(index->update(key_array[i], reinterpret_cast<P>(&key_array[i]))){
          updates++;
        }
      }
    }

    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during update = " << num - updates << std::endl;
    end_notify();
}

template <class T, class P>
void update_without_epoch(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    T *key_array = reinterpret_cast<T *>(_range->workload);
    uint64_t updates = 0;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < num; ++i) {
      if(index->update(key_array[i], reinterpret_cast<P>(&key_array[i]))){
        updates++;
      }
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Not found during update = " << num - updates << std::endl;
    end_notify();
}

/*** Benchmark Mixed Operation ***/

// Mixed Insert and Search operations
template <class T, class P>
void concurr_mixed(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    typedef std::pair<int, T> OPT;
    OPT *key_array = reinterpret_cast<OPT *>(_range->workload);
    uint64_t round = num / EPOCH_DURATION;
    uint64_t i = 0;
    uint64_t found = 0;
    P payload;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    while (i < round) {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      uint64_t end = (i + 1) * EPOCH_DURATION;
      for (uint64_t j = i * EPOCH_DURATION; j < end; ++j) {
        auto op = key_array[j].first;
        auto key = key_array[j].second;
        if(op == 0){
          index->insert(key, reinterpret_cast<P>(&(key_array[j].second)));
        }else{
          found += index->search(key, &payload);
        }
      }
      ++i;
    }

    {
      auto epoch_guard = my_alloc::BasePMPool::AquireEpochGuard();
      for (i = EPOCH_DURATION * round; i < num; ++i) {
        auto op = key_array[i].first;
        auto key = key_array[i].second;
        if(op == 0){
          index->insert(key, reinterpret_cast<P>(&(key_array[i].second)));
        }else{
          found += index->search(key, &payload);
        }
      }
    }

    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Found during mixed = " << found << std::endl;
    end_notify();
}

template <class T, class P>
void mixed_without_epoch(Tree<T, P> *index, struct Range *_range) {
    set_affinity(_range->id);
    uint64_t num = _range->num;
    typedef std::pair<int, T> OPT;
    OPT *key_array = reinterpret_cast<OPT *>(_range->workload);
    uint64_t found = 0;
    P payload;

    spin_wait();
    auto workload_start_time = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < num; ++i) {
      auto op = key_array[i].first;
      auto key = key_array[i].second;
      if(op == 0){
        index->insert(key, reinterpret_cast<P>(&(key_array[i].second)));
      }else{
        found += index->search(key, &payload);
      }
    }
    
    auto workload_end_time = std::chrono::high_resolution_clock::now();
    _range->total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(workload_end_time -
                                                            workload_start_time)
          .count();
    std::cout << "Found during mixed = " << found << std::endl;
    end_notify();
}