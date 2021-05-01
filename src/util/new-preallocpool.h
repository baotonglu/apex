#pragma once

#include <iostream>
#include <cstddef>
#include <climits>
#include <cstdlib>
#include <new>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <cmath>
#include <cstring>
#include <tuple>
#include <thread>

#include "libpmem.h"
#include "libpmemobj.h"
#include "allocator.h"
#include "utils.h"

#define TEST_PRE 1

// Preallocator must be initiliazed after the initialize of the memory allocator
namespace my_alloc{
// CHUNK_SIZE is 4MB
#define CHUNK_SIZE 2*1024*1024    

    class allocate_log{
        public:
        void **alloc_ptr;
        void *allocate_chunk;
        void *allocate_object;
        size_t free_num;
        int in_use;//1 means allocate, 01 means deallocate
    };
    
    // Allocate one memory chunk a time to amortize the allocation overhead
    class memory_chunk{
        public: 
        int free_num_; // free num is used to determine the number of free nodes
        int in_array_;
        memory_chunk *next_; // link next memory chunck
        memory_chunk *prev_; // link previous memory chunck
        void *free_nodes_; // used to link the free memory nodes, instead of using bitmap to mamange the records
        size_t allocate_index_;
        int total_num_;
        int lock_;
        char alloc_space[1]; // for pre-allocation

        void initialize(memory_chunk *next, memory_chunk *prev, int total_num){
            next_ = next;
            prev_ = prev;
            free_num_ = 0;
            free_nodes_ = nullptr;
            allocate_index_ = 0;
            total_num_ = total_num;
            in_array_ = 0;
            lock_ = 0;
        }

        // @total_num: #blocks in this chunk
        // @alloc_size: base alloc size in one chunk
        static void New(PMEMoid *dir, memory_chunk *next, memory_chunk *prev, int total_num, int alloc_size){
            auto callback = [](PMEMobjpool *pool, void *ptr, void *arg) {
                auto value_ptr = reinterpret_cast<std::tuple<memory_chunk*, memory_chunk*, int, int>*>(arg);
                auto chunk_ptr = reinterpret_cast<memory_chunk *>(ptr);
                memset(chunk_ptr, 0, CHUNK_SIZE);// first clear zero
                chunk_ptr->next_ = std::get<0>(*value_ptr);
                chunk_ptr->prev_ = std::get<1>(*value_ptr);
                chunk_ptr->total_num_ = std::get<2>(*value_ptr);
                int alloc_size = std::get<3>(*value_ptr); // set the data according to the alloc_size
                chunk_ptr->free_nodes_ = reinterpret_cast<void*>(chunk_ptr->alloc_space + 8);
                chunk_ptr->allocate_index_ = 0;
                int total_num = chunk_ptr->total_num_;
                chunk_ptr->free_num_ = total_num;
                for(int i = 0; i < total_num - 1; ++i){
                    uint64_t *shift = reinterpret_cast<uint64_t*>(chunk_ptr->alloc_space + i * alloc_size);
                    *shift = i;
                    void** link_addr = reinterpret_cast<void**>(chunk_ptr->alloc_space + i * alloc_size + 8);
                    *link_addr = reinterpret_cast<void*>(chunk_ptr->alloc_space + (i+1) * alloc_size + 8);
                }
                uint64_t *shift = reinterpret_cast<uint64_t*>(chunk_ptr->alloc_space + (total_num - 1) * alloc_size);
                *shift = total_num - 1;
                clwbmore(ptr,  ptr + CHUNK_SIZE);
                //pmemobj_persist(pool, chunk_ptr, CHUNK_SIZE);
                return 0;
            };

            std::tuple<memory_chunk*, memory_chunk*, int, int> callback_args = {next, prev, total_num, alloc_size};
            BasePMPool::Allocate(dir, 64, CHUNK_SIZE, callback, reinterpret_cast<void *>(&callback_args));
        }

        bool is_full(){
            if(free_num_ == 0){
                return true;
            }
            return false;    
        }

        bool is_empty(){
            if(free_num_ == total_num_){
                return true;
            }
            return false;
        }

        void* allocate_one_with_persistence(void**ptr, size_t alloc_size, allocate_log *log){
            // Reuse free nodes
            void *allocate_node;
            log->alloc_ptr = ptr;
            allocate_node = reinterpret_cast<void*>(free_nodes_);
            log->allocate_object = reinterpret_cast<void*>(free_nodes_);
            log->in_use = 1;
            log->free_num = free_num_;

            clwb(log);
            sfence(); // indicate that our allocation starts

            free_nodes_ = *reinterpret_cast<void**>(free_nodes_);

            // clear the link addr
            *reinterpret_cast<uint64_t*>(log->allocate_object) = 0;
            //clwb(log->allocate_object);

            free_num_--;
            clwb(&free_num_); //flush both free_num_ and free_nodes

            *ptr = log->allocate_object; // attach the object to data structure
            clwb(ptr);
            sfence(); // Then the allocation finishes

            log->in_use = 0;
            memset(log, 0, sizeof(allocate_log));
            clwb(log); // Flush, no fence is used since next operation must has one fence
            return allocate_node;
        }

        void* allocate_one_with_persistence(void**ptr, size_t alloc_size, allocate_log *log, int (*alloc_constr)(void* ptr,
                                        void* arg), void* arg){
            // Reuse free nodes
            void *allocate_node;
            log->alloc_ptr = ptr;
            allocate_node = reinterpret_cast<void*>(free_nodes_);
            log->allocate_object = reinterpret_cast<void*>(free_nodes_);
            log->in_use = true;
            log->free_num = free_num_;

            clwb(log);
            sfence(); // indicate that our allocation starts

            free_nodes_ = *reinterpret_cast<void**>(free_nodes_);

            *reinterpret_cast<uint64_t*>(log->allocate_object) = 0;
            alloc_constr(log->allocate_object, arg);

            free_num_--;
            clwb(&free_num_); //flush both free_num_ and free_nodes

            *ptr = log->allocate_object; // attach the object to data structure
            clwb(ptr);
            sfence(); // Then the allocation finishes

            log->in_use = false;
            memset(log, 0, sizeof(allocate_log));
            clwb(log); // Flush, no fence is used since next operation must has one fence
            return allocate_node;
        }

        void print_free_nodes(){
            auto node = free_nodes_;
            while(node != nullptr){
                //printf("free addr is %p\n", node);
                node = *((void**)node);
            }
        }   
    };
    

    // The class that manages the memory chunks pre-allocated
    // Not crash consitent, need FIX
    class free_array{
        public:
        size_t number_; 
        size_t total_num_;
        memory_chunk *free_memory_[1];

        void initialize(size_t total_num, free_array *other = nullptr){
            if(other){
                total_num_ = total_num;
                number_ = other->number_;
                memcpy(free_memory_, other->free_memory_, sizeof(memory_chunk*)*other->total_num_);
            }else{
                number_ = 0;
                total_num_ = total_num;
            }
        }

        bool add(memory_chunk * new_chunk){
            //printf("add a new chunk %p\n", new_chunk);
            if(number_ == total_num_){
                return false; // Need expand the array
            }
            free_memory_[number_] = new_chunk;
            new_chunk->in_array_ = 1;
            number_++;
            return true;
        }

        void remove(memory_chunk *old_chunk){
            if(number_ != 0){
                for(int i = 0; i < number_; i++){
                    if(free_memory_[i] == old_chunk){
                        for(int j = i; j < number_ - 1; j++){
                            free_memory_[j] = free_memory_[j + 1];
                        }
                        number_--;
                        old_chunk->in_array_ = 0;
                        //std::cout << "Remove the chunk at " << i << std::endl;
                        return;
                    }
                }
            }
        }
    };

    /// Why using static variable in these variables ?
    /// Actually, the static variables of this class is in DRAM, lost after system failure.
    /// For correct recovery, I should use serveral instances of PreAllocPool
    class PreAllocPool{
    public:
        PMEMoid p_pre_alloc_memory_;
        PMEMoid p_new_chunk_;
        PMEMoid p_free_array_; // PMDK Allocation of free array in this 16-byte pointer
        memory_chunk* pre_alloc_memory_;// A linked list to manage all of the pre-allocated memory
        free_array* free_array_; // An array which manages the memory chunk which has free memory to use
        size_t base_alloc_size_; // Every time allocated the same chunk size
        allocate_log* log_;
        uint64_t in_use; // used to indicate whether this memory pool is in use or free
        char dummy[40]; // for cacheline alignment

        void init(size_t base_alloc_size){ 
            // It should be 256-byte when allocating the small overflow chunk        
            base_alloc_size_ = static_cast<size_t>(std::ceil((base_alloc_size + 8) / 64.) * 64); 
            BasePMPool::ZAllocate(&p_free_array_, offsetof(free_array, free_memory_) + sizeof(memory_chunk*) * 1024);
            free_array_ = reinterpret_cast<free_array*>(pmemobj_direct(p_free_array_));

            free_array_->initialize(1024);  
            //BasePMPool::ZAllocate(&p_pre_alloc_memory_, CHUNK_SIZE);
            //pre_alloc_memory_ = reinterpret_cast<memory_chunk*>(pmemobj_direct(p_pre_alloc_memory_));
            //pre_alloc_memory_->initialize(nullptr, nullptr, (CHUNK_SIZE-48) / base_alloc_size_);
            memory_chunk::New(&p_pre_alloc_memory_, nullptr, nullptr, (CHUNK_SIZE-48) / base_alloc_size_, base_alloc_size_);
            pre_alloc_memory_ = reinterpret_cast<memory_chunk*>(pmemobj_direct(p_pre_alloc_memory_));
            // Only add it when there is node free
            free_array_->add(pre_alloc_memory_); 
            p_new_chunk_ = OID_NULL;
            BasePMPool::ZAllocate((void**)&log_, sizeof(allocate_log));
        } 

        //Allocate the node in a crash consistency way, no initialization function provided
        void* alloc_node_with_persistence(void **ptr){
            auto free_num = free_array_->number_;
            if(free_num != 0){
                //directly use free node
                memory_chunk *free_chunk = free_array_->free_memory_[free_num - 1];
                log_->allocate_chunk = free_chunk;

                int old_value = 0;
                int new_value = 1;
                while(!CAS(&free_chunk->lock_, &old_value, new_value)){
                    old_value = 0;
                }
                auto my_node = free_chunk->allocate_one_with_persistence(ptr, base_alloc_size_, log_);
                if(free_chunk->is_full()){
                    free_chunk->in_array_ = 0;
                    free_array_->number_--;
                }

                STORE(&free_chunk->lock_, 0);
                return my_node;
            }
            // perfBasePMPool::ZAllocate(&p_new_chunk_, CHUNK_SIZE);
            memory_chunk::New(&p_new_chunk_, pre_alloc_memory_, nullptr, (CHUNK_SIZE-48)/base_alloc_size_, base_alloc_size_);
            auto new_chunk = reinterpret_cast<memory_chunk*>(pmemobj_direct(p_new_chunk_));
            //new_chunk->initialize(pre_alloc_memory_,nullptr,(CHUNK_SIZE-48)/base_alloc_size_);
            if(pre_alloc_memory_ != nullptr){
                pre_alloc_memory_->prev_ = new_chunk;
            }
            pre_alloc_memory_ = new_chunk;
            auto my_node = new_chunk->allocate_one_with_persistence(ptr, base_alloc_size_, log_);
            free_array_->add(new_chunk);
            /*
            if(new_chunk->is_full()){
                new_chunk->in_array_ = 0;
                free_array_->number_--;
            }*/
            return my_node;
        }

        //Allocate the node in a crash consistency way
        void* alloc_node_with_persistence(void **ptr,int (*alloc_constr)(void* ptr,
                                           void* arg), void* arg){
            auto free_num = free_array_->number_;
            if(free_num != 0){
                //directly use free node
                memory_chunk *free_chunk = free_array_->free_memory_[free_num - 1];
                log_->allocate_chunk = free_chunk;
                //first get the lock of this chunk
                int old_value = 0;
                int new_value = 1;
                while(!CAS(&free_chunk->lock_, &old_value, new_value)){
                    old_value = 0;
                }

                auto my_node = free_chunk->allocate_one_with_persistence(ptr, base_alloc_size_, log_, alloc_constr, arg);
                if(free_chunk->is_full()){
                    free_chunk->in_array_ = 0;
                    free_array_->number_--;
                }

                STORE(&free_chunk->lock_, 0);
                return my_node;
            }
            // Allocate more chunks
            memory_chunk::New(&p_new_chunk_, pre_alloc_memory_, nullptr, (CHUNK_SIZE-48)/base_alloc_size_, base_alloc_size_);
            auto new_chunk = reinterpret_cast<memory_chunk*>(pmemobj_direct(p_new_chunk_));
            if(pre_alloc_memory_ != nullptr){
                pre_alloc_memory_->prev_ = new_chunk;
            }
            pre_alloc_memory_ = new_chunk;
            auto my_node = new_chunk->allocate_one_with_persistence(ptr, base_alloc_size_, log_, alloc_constr, arg);
            free_array_->add(new_chunk);
            /*
            if(new_chunk->is_full()){
                new_chunk->in_array_ = 0;
                free_array_->number_--;
            }*/
            return my_node;
        }

        void free_node(void **node_ptr){
            void *node = *node_ptr;
            uint64_t* size_shift = reinterpret_cast<uint64_t*>((char*)node - 8);
            char *cur_addr = reinterpret_cast<char*>(size_shift);
            memset(node, 0, base_alloc_size_ - 8); //use memset to reset the segment
            memory_chunk* target_chunk = reinterpret_cast<memory_chunk*>(cur_addr - (*size_shift) * base_alloc_size_ - 48);
            
            //first get the lock of this chunk
            int old_value = 0;
            int new_value = 1;
            while(!CAS(&target_chunk->lock_, &old_value, new_value)){
                old_value = 0;
            }

            // First record in the log
            log_->allocate_chunk = target_chunk;
            log_->allocate_object = node;
            log_->free_num = target_chunk->free_num_;
            log_->alloc_ptr = node_ptr;
            log_->in_use = -1;

            // Flush the log to memory
            clwb(log_);
            sfence();

            // try to link
            void** ptr = reinterpret_cast<void**>(node);
            *ptr = target_chunk->free_nodes_;

            // Flush the contents in freed memory
            clwbmore(node, reinterpret_cast<char*>(node) + base_alloc_size_ - 8); 

            target_chunk->free_nodes_ = node; // free nodes points to the real node
            target_chunk->free_num_++;

            *node_ptr = nullptr;

            // flush the node_ptr in data structure
            clwb(node_ptr); 

            // flush the metadata in group allcoator
            clwb(&target_chunk->free_nodes_);
            sfence();

            // reset the log
            log_->in_use = 0;
            memset(log_, 0, sizeof(allocate_log));
            clwb(log_); 

            // The chunk is only added to the free array when its load factor drops by 50%
            if((target_chunk->free_num_ == (target_chunk->total_num_ / 2)) && (target_chunk->in_array_ == 0)){
                //Need to expand the free array
                if(!free_array_->add(target_chunk)){
                    auto old_array = free_array_;
                    BasePMPool::ZAllocate((void**)&free_array_, offsetof(free_array, free_array::free_memory_) + sizeof(memory_chunk*) * old_array->total_num_ * 2);
                    free_array_->initialize(old_array->total_num_ * 2, old_array);
                    BasePMPool::Free(old_array);
                    free_array_->add(target_chunk);
                }
                //std::cout << "add to the array" << std::endl;
            }

            STORE(&target_chunk->lock_, 0);
        }

        void display_info(){
            auto cur_mem = pre_alloc_memory_;
            int num;
            memory_chunk *pre_chunk = nullptr;
            while(cur_mem){
                //printf("-- The addr is %p\n", cur_mem);
                num++;
                //std::cout << "Allocate index is " << cur_mem->allocate_index_ << "; total_num is " << cur_mem->total_num_ << std::endl;
                pre_chunk = cur_mem;
                cur_mem = cur_mem->next_;
            }

            while(pre_chunk){
                cur_mem = pre_chunk;
                pre_chunk = cur_mem->prev_;
            }
            //printf("The initial addr is %p", cur_mem);
        }
    };

    // An array of prealloc-pool 
    // Every thread get one instance from this class
    class PoolArray{
        public:
        static PMEMoid p_pool_array_;
        static PreAllocPool* pool_array_;
        static int array_size_;

        static void initialize(int size){
            //std::cout << "The size of preallocpool is " << sizeof(PreAllocPool) << std::endl;
            array_size_ = size;
            BasePMPool::ZAllocate(&p_pool_array_, sizeof(PreAllocPool) * size);
            pool_array_ = reinterpret_cast<PreAllocPool*>(pmemobj_direct(p_pool_array_));
        }

        // Find the existing memory pool or assign a new memory pool to this thread
        static PreAllocPool* assign_pool(uint64_t* assign_index){
            uint64_t current_thread_id = pthread_self();
            // first scan to see whether I already has a memory pool registerd
            for(int i = 0; i < array_size_; ++i){
                if(pool_array_[i].in_use == current_thread_id){
                    *assign_index = i;
                    return &(pool_array_[i]);
                }
            }

            // regitster the thread id in this pool
            while(true){
                for(int i = 0; i < array_size_; ++i){
                    uint64_t old_value = 0;
                    uint64_t new_value = current_thread_id;
                    if(CAS(&pool_array_[i].in_use, &old_value, new_value)){
                        *assign_index = i;
                        // The pool should be firstly initialized
                        pool_array_[i].init(256); // This is the block size of block
                        return &(pool_array_[i]);
                    }
                }
            }
        }

        static void return_pool(uint64_t assign_index){
            uint64_t current_thread_id = pthread_self();
            if(pool_array_[assign_index].in_use == current_thread_id){
                pool_array_[assign_index].in_use = 0;
            }
        }
    };

    PMEMoid PoolArray::p_pool_array_ = OID_NULL;
    PreAllocPool* PoolArray::pool_array_ = nullptr;
    int PoolArray::array_size_ = 0;
    // PMEMoid PreAllocPool::p_pre_alloc_memory_ = OID_NULL;
    // PMEMoid PreAllocPool::p_new_chunk_ = OID_NULL;
    // memory_chunk* PreAllocPool::pre_alloc_memory_ = nullptr;// A linked list to manage all of the pre-allocated memory
    // PMEMoid PreAllocPool::p_free_array_ = OID_NULL;
    // free_array* PreAllocPool::free_array_ = nullptr; // An array which manage the memory chunk which has free memory to use
    // size_t PreAllocPool::base_alloc_size_ = 0; // Every time allocated the same chunk size
    // allocate_log* PreAllocPool::log_ = nullptr;
}