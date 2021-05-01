// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Simple benchmark that runs a mixture of point lookups and inserts on ALEX.
 */

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

/*
 * Required flags:
 * --keys_file              path to the file that contains keys
 * --keys_file_type         file type of keys_file (options: binary or text)
 * --init_num_keys          number of keys to bulk load with
 * --total_num_keys         total number of keys in the keys file
 * --batch_size             number of operations (lookup or insert) per batch
 *
 * Optional flags:
 * --insert_frac            fraction of operations that are inserts (instead of
 * lookups)
 * --lookup_distribution    lookup keys distribution (options: uniform or zipf)
 * --time_limit             time limit, in minutes
 * --print_batch_stats      whether to output stats for each batch
 */

// Global parameters
std::string keys_file_path;
std::string keys_file_type;
std::string keys_type;
int init_num_keys;
int init_insert_keys;
int workload_keys; // Number of keys to operation in the workload
int total_num_keys;
int batch_size;
int sleep_ms; // The sleep for each thread
std::string operation;
double insert_frac;
std::string lookup_distribution;
bool using_epoch = false;
bool skip_bulkload = false;
bool random_shuffle = false;
bool sort_bulkload = true;
int thread_num;
std::string index_type;
double recover_time;
double theta;

void get_SMO_info(){ 
  std::cout << "reisze_without_decision = " << alex::resize_without_decision << std::endl;
  std::cout << "reisze_with_decision = " << alex::resize_with_decision << std::endl;
  std::cout << "split_sideway_times = " << alex::split_times << std::endl;
  std::cout << "cata_cost times = " << alex::cata_cost_times << std::endl;
  std::cout << "deviation_times = " << alex::deviation_times << std::endl;
}

template<class T, class P>
Tree<T, P>* generate_index(){
  Tree<T, P> *index = nullptr;
  auto start_time = std::chrono::high_resolution_clock::now();
  if(index_type == "alex"){
    bool recover = my_alloc::BasePMPool::Initialize(pool_name, pool_size);
    auto index_ptr = reinterpret_cast<Tree<T, P> **>(my_alloc::BasePMPool::GetRoot(sizeof(Tree<T, P>*)));
    if (recover)
    {
      index = reinterpret_cast<Tree<T, P>*>(reinterpret_cast<char*>(*index_ptr) + 48);
      new (index) alex::Alex<T, P>(recover);
    }else{ 
      my_alloc::BasePMPool::ZAllocate(reinterpret_cast<void**>(index_ptr), sizeof(alex::Alex<T, P>) + 64);
      index = reinterpret_cast<Tree<T, P>*>(reinterpret_cast<char*>(*index_ptr) + 48);     
      new (index) alex::Alex<T, P>();
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

// In this benchmark, runtime throughput is recorded
template <class T, class P>
void ProfileBench(Tree<T, P> *index, Range *rarray, int thread_num, void (*test_func)(Tree<T, P> *, struct Range *)) {
  std::thread *thread_array[1024];
  double duration;
  finished = false;
  bar_a = 1;
  bar_b = thread_num;
  bar_c = thread_num;
  uint64_t *last_record = new uint64_t[thread_num];
  uint64_t *curr_record = new uint64_t[thread_num];
  memset(last_record, 0, sizeof(uint64_t) * thread_num);
  memset(curr_record, 0, sizeof(uint64_t) * thread_num);
  double seconds = (double)sleep_ms / 1000;

  for (uint64_t i = 0; i < thread_num; ++i) {
    thread_array[i] = new std::thread(*test_func, index, &rarray[i]);
  }

  while (LOAD(&bar_b) != 0)
    ;                                     // Spin
  std::unique_lock<std::mutex> lck(mtx);  // get the lock of condition variable

  STORE(&bar_a, 0);  // start test
  // Start to do the sampling and record in the file
  while (bar_c != 0) {
    msleep(sleep_ms); // sleep 0.1s
    for (int i = 0; i < thread_num; ++i) {
      curr_record[i] = operation_record[i].number;
    }
    uint64_t operation_num = 0;
    for (int i = 0; i < thread_num; ++i) {
      operation_num += (curr_record[i] - last_record[i]);
    }
    double throughput = operation_num / 1000000.0 / seconds;
    std::cout << throughput << std::endl; /*Mops/s*/
    memcpy(last_record, curr_record, sizeof(uint64_t) * thread_num);
  }

  for (int i = 0; i < thread_num; ++i) {
    thread_array[i]->join();
    delete thread_array[i];
  }

  // Directly collect the throughput of each thread and accumulate them
  double total_throughput = 0;
  for (int i = 0; i < thread_num; i++)
  {
    total_throughput += rarray[i].num / rarray[i].total_time * 1e9;
  }

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
    //std::cout << "Min key in the data set = " << my_keys[0] << std::endl;
    //std::cout << "Max key in the data set = " << my_keys[total_num_keys - 1] << std::endl;
    // Copy from vector to keys and then shuffle it
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
  int num_pre_insertion = i + init_insert_keys;

  // Insert keys into index
  std::cout << "Start the pre-insertion" << std::endl;
  for (; i < num_pre_insertion; i++) {
    index->insert(keys[i], (PAYLOAD_TYPE)(&keys[i]));
  }
  std::cout << "End the pre-insertition" << std::endl;

  // Generate the workload
  // Mixed workload (mixing search and insert), search/erase/update workload (existence keys), insert workload (new keys)
  uint64_t generate_num = workload_keys;
  char *workload =  generate_workload(operation, generate_num, keys, num_pre_insertion, insert_frac, batch_size, lookup_distribution, theta);
  //delete [] keys;

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
    } else if (operation == "range_debug"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &range_debug);
    } else if (operation == "recovery"){
      ProfileBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_search_with_sample);
    } else {
      // Full evaluation of search, range query, update operation since they could use the same workload
      // 1. Search
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_search);

      // 2. Range query
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_range);

      // 3. Update
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &concurr_update);
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
    } else if (operation == "range_debug"){
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &range_debug);
    } else {
      // 1. Search
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &search_without_epoch);

      // 2. Range query
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &range_without_epoch);

      // 3. Update
      GeneralBench<T, PAYLOAD_TYPE>(index, range_array, thread_num, &update_without_epoch);
    }       
  }

  index->get_depth_info();
  //get_SMO_info();
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
  init_insert_keys = stoi(get_required(flags, "init_insert_keys"));
  workload_keys = stoi(get_required(flags, "workload_keys")); // Number of operations in the workload
  total_num_keys = stoi(get_required(flags, "total_num_keys"));
  batch_size = stoi(get_required(flags, "batch_size"));
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
  sleep_ms = stoi(get_required(flags, "sleep_ms"));

  if(using_epoch){
    std::cout << "The epoch is used" << std::endl;
  }
  std::cout << "The key type is " << keys_type << std::endl;

  if(keys_type == "double"){
    Run<double>();
  }else{
    Run<uint64_t>(); // Other keys are regarded as uint64_t
  }  
  
  return 0;  
}