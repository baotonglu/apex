// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//#define PROFILE 1
#include "../core/apex.h"
#include "../util/System.h"
#include "../util/sosd_util.h"
#include "operation.h"
#include <sys/time.h>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include "flags.h"
#include "utils.h"

// Modify these if running your own workload
#define PAYLOAD_TYPE char*

// Global parameters
std::string keys_file_path;
std::string keys_file_type;
std::string keys_type;
int init_num_keys;
int workload_keys; // Number of keys to operation in the workload
int total_num_keys;
std::string operation;
double insert_frac;
std::string lookup_distribution;
bool using_epoch = false;
bool skip_bulkload = false;
bool random_shuffle = false;
bool sort_bulkload = true;
int thread_num;
std::string index_type;
double theta;
int batch_size = 10000000;

template<class T, class P>
Tree<T, P>* generate_index(){
  Tree<T, P> *index = nullptr;
  auto start_time = std::chrono::high_resolution_clock::now();
  if(index_type == "apex"){
    bool recover = my_alloc::BasePMPool::Initialize(pool_name, pool_size);
    auto index_ptr = reinterpret_cast<Tree<T, P> **>(my_alloc::BasePMPool::GetRoot(sizeof(Tree<T, P>*)));
    if (recover)
    {
      index = reinterpret_cast<Tree<T, P>*>(reinterpret_cast<char*>(*index_ptr) + 48);
      new (index) alex::Apex<T, P>(recover);
    }else{ 
      my_alloc::BasePMPool::ZAllocate(reinterpret_cast<void**>(index_ptr), sizeof(alex::Apex<T, P>) + 64);
      index = reinterpret_cast<Tree<T, P>*>(reinterpret_cast<char*>(*index_ptr) + 48);     
      new (index) alex::Apex<T, P>();
    }
  }else{
    std::cerr << "No index is matched." << std::endl;
    exit(-1);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  double consume_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                          start_time)
          .count();
  std::cout << "Recover/Initialize time (ms) = " << consume_time / (1e6) << std::endl;
  return index;
}

// Benchmark
template <class T, class P>
void GeneralBench(Tree<T, P> *index, Range *rarray, int thread_num, void (*test_func)(Tree<T, P> *, struct Range *)) {
  std::thread *thread_array[1024];
  double duration;
  finished = false;
  bar_a = 1;
  bar_b = thread_num;
  bar_c = thread_num;

  std::string profile_name = "alex_profile";
  //System::profile(profile_name, [&]() {

  for (uint64_t i = 0; i < thread_num; ++i) {
    thread_array[i] = new std::thread(*test_func, index, &rarray[i]);
  }

  while (LOAD(&bar_b) != 0)
    ;                                     // Spin
  std::unique_lock<std::mutex> lck(mtx);  // get the lock of condition variable

  STORE(&bar_a, 0);  // start test
  while (!finished) {
    cv.wait(lck);  // go to sleep and wait for the wake-up from child threads
  }

  for (int i = 0; i < thread_num; ++i) {
    thread_array[i]->join();
    delete thread_array[i];
  }

  //});

  double total_throughput = 0;
  double longest_time = 0;
  uint64_t total_num = 0;
  for (int i = 0; i < thread_num; i++)
  {
    if(longest_time < rarray[i].total_time){
      longest_time = rarray[i].total_time;
    }
    total_num += rarray[i].num;
  }

  total_throughput += total_num / longest_time * 1e9;

  std::cout << "\tcumulative throughput:\t"
            << total_throughput << " ops/sec"
            << std::endl;
}


// Run function, to select the workload to run
template <class T>
void Run(){
  // Read keys from file
  T* keys = new T[total_num_keys];
  if (keys_file_type == "binary") {
    load_binary_data(keys, total_num_keys, keys_file_path);
  } else if (keys_file_type == "text") {
    load_text_data(keys, total_num_keys, keys_file_path);
  } else if (keys_file_type == "sosd") {
    // Benchmark on SOSD data, using SOSD's loading method
    std::vector<T> my_keys = util::load_data<T>(keys_file_path);
    bool unique_keys_ = util::is_unique<T>(my_keys);
    if (unique_keys_)
      std::cout << "data is unique" << std::endl;
    else
      std::cout << "data contains duplicates" << std::endl;
    T* copy_keys = &my_keys[0];
    memcpy(keys, copy_keys, sizeof(T) * total_num_keys);
    random_shuffle = true;
  } else {
    std::cerr << "--keys_file_type must be either 'binary' or 'text'"
              << std::endl;
    return;
  }

  if(random_shuffle){
    std::random_shuffle(&keys[0], &keys[total_num_keys - 1]);
  }

  // Combine bulk loaded keys with generated values
  auto values = new std::pair<T, PAYLOAD_TYPE>[init_num_keys];
  std::mt19937_64 gen_payload(std::random_device{}());
  for (int i = 0; i < init_num_keys; i++) {
    values[i].first = keys[i];
    values[i].second = reinterpret_cast<PAYLOAD_TYPE>(&keys[i]);
  }

  Tree<T, PAYLOAD_TYPE> *index = generate_index<T, PAYLOAD_TYPE>();

  // Bulk loading keys to the index
  if(sort_bulkload){
     std::sort(values, values + init_num_keys,
            [](auto const& a, auto const& b) { return a.first < b.first; });
  }

  if(!skip_bulkload){
    std::cout << "Start the bulk load" << std::endl;
    std::cout << "The min key = " << values[0].first << std::endl;
    std::cout << "The max key = " << values[init_num_keys - 1].first << std::endl;
    index->bulk_load(values, init_num_keys);
    delete [] values;   
    std::cout << "End the bulk load" << std::endl;
    index->get_depth_info();
  }

  int i = init_num_keys;
  int num_pre_insertion = i;

  // Generate the workload
  // Mixed workload (mixing search and insert), search/erase/update workload (existence keys), insert workload (new keys)
  uint64_t generate_num = workload_keys;
  char *workload =  generate_workload(operation, generate_num, keys, num_pre_insertion, insert_frac, batch_size, lookup_distribution, theta);

  // Partition the workload
  Range *range_array = new Range[thread_num];
  if(operation == "mixed"){
    // Benchmark mixed operation
    typedef std::pair<int, T> OPT;
    OPT *my_workload = reinterpret_cast<OPT *>(workload);
    auto partition_num = generate_num / thread_num;
    for(int i = 0; i < (thread_num - 1); ++i){
      range_array[i].id = 2 * i;
      range_array[i].num = partition_num;
      range_array[i].workload = reinterpret_cast<char*>(my_workload + i * partition_num);
    }

    range_array[thread_num - 1].id = 2 * (thread_num - 1);
    range_array[thread_num - 1].num = generate_num - partition_num * (thread_num - 1);
    range_array[thread_num - 1].workload = reinterpret_cast<char*>(my_workload + (thread_num - 1) * partition_num);
  }else{
    // Benchmark single operation
    T *my_workload = reinterpret_cast<T*>(workload);
    auto partition_num = generate_num / thread_num;
    for(int i = 0; i < (thread_num - 1); ++i){
      range_array[i].id = 2 * i;
      range_array[i].num = partition_num;
      range_array[i].workload = reinterpret_cast<char*>(my_workload + i * partition_num);
    }

    range_array[thread_num - 1].id = 2 * (thread_num - 1);
    range_array[thread_num - 1].num = generate_num - partition_num * (thread_num - 1);
    range_array[thread_num - 1].workload = reinterpret_cast<char*>(my_workload + (thread_num - 1) * partition_num);
  }

  // Run the benchmark
  if(using_epoch == true){
    if(operation == "mixed"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_mixed);
    } else if (operation == "insert"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_insert);
    } else if (operation == "search"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_search);
    } else if (operation == "erase"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_erase);
    } else if (operation == "update"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_update);
    } else if (operation == "range"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_range);
    } else {
      std::cout << "Unknown operation " << operation << std::endl;
    }    
  }else{
    if(operation == "mixed"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &mixed_without_epoch);
    } else if (operation == "insert"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &insert_without_epoch);
    } else if (operation == "search"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &search_without_epoch);
    } else if (operation == "erase"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &erase_without_epoch);
    } else if (operation == "update"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &update_without_epoch);
    } else if (operation == "range"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &range_without_epoch);
    } else {
      std::cout << "Unknown operation " << operation << std::endl;
    }       
  }

  delete [] keys;
  my_alloc::BasePMPool::ClosePool();
}

int main(int argc, char* argv[]) {
  set_affinity(0);

  // Get the flag from user
  auto flags = parse_flags(argc, argv);
  keys_file_path = get_required(flags, "keys_file");
  keys_file_type = get_required(flags, "keys_file_type");
  keys_type = get_required(flags, "keys_type");
  init_num_keys = stoi(get_required(flags, "init_num_keys"));
  workload_keys = stoi(get_required(flags, "workload_keys")); // Number of operations in the workload
  total_num_keys = stoi(get_required(flags, "total_num_keys"));
  operation = get_required(flags, "operation"); // Which operation to evalaute
  insert_frac = stod(get_with_default(flags, "insert_frac", "0.5"));
  theta = stod(get_with_default(flags, "theta", "0.99"));
  lookup_distribution =
      get_with_default(flags, "lookup_distribution", "zipf");
  int epoch_flag = stoi(get_required(flags, "using_epoch"));
  if(epoch_flag){
    using_epoch = true;
  }else{
    using_epoch = false;
  }
  random_shuffle = get_boolean_flag(flags, "random_shuffle");
  int sort_flag = stoi(get_required(flags, "sort_bulkload"));
  if (sort_flag)
  {
    sort_bulkload = true;
  }else{
    sort_bulkload = false;
  }
  skip_bulkload = get_boolean_flag(flags, "skip_bulkload");
  thread_num = stoi(get_with_default(flags, "thread_num", "1"));
  index_type = get_required(flags, "index");

  // Print some critical information
  std::cout << "The key type is " << keys_type << std::endl;
  if(using_epoch){
    std::cout << "The epoch is used" << std::endl;
  }  

  if(keys_type == "double"){
    Run<double>();
  }else{
    Run<uint64_t>(); // Other keys are regarded as uint64_t
  }  
  return 0;  
}