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

//#define  DRAM_ALLOC 1

#ifndef DRAM_ALLOC
#include "libpmem.h"
#include "libpmemobj.h"
#include "allocator.h"
#endif

//#define FLUSH 1
#define TEST_PRE 1

// Preallocator must be initiliazed after the initialize of the memory allocator
namespace my_alloc{
// CHUNK_SIZE is 2MB
#define CHUNK_SIZE 16*1024*1024    

    class allocate_log{
            public:
            void *allocate_object;
        };

        class memory_chunk{
            public: 
            size_t free_num_; // free num is used to determine the number of free nodes
            memory_chunk *next_; // link next memory chunck
            memory_chunk *prev_; // link previous memory chunck
            void *free_nodes_; // used to link the free memory nodes, instead of using bitmap to mamange the records
            size_t allocate_index_;
            size_t total_num_;
            char alloc_space[1]; // for pre-allocation

            void initialize(memory_chunk *next, memory_chunk *prev, size_t total_num){
                next_ = next;
                prev_ = prev;
                free_num_ = 0;
                free_nodes_ = nullptr;
                allocate_index_ = 0;
                total_num_ = total_num;
            }

            bool is_full(){
                if((allocate_index_ == total_num_) && (free_num_ == 0)){
                    return true;
                }
                return false;    
            }

            bool is_empty(){
                if((allocate_index_ == total_num_) && (free_num_ == total_num_)){
                    return true;
                }
                return false;
            }

            void allocate_one(size_t alloc_size, allocate_log *log){
                if(free_nodes_ != nullptr){
                    // Reuse free nodes
                    log->allocate_object = reinterpret_cast<void*>(free_nodes_);
                    free_nodes_ = *reinterpret_cast<void**>(free_nodes_);
                    memset(log->allocate_object, 0, 8);// clear the next pointer
#ifdef FLUSH
                    clwb(log->allocate_object);
                    sfence();
#endif
                    free_num_--;
#ifdef FLUSH
                    clwb(&free_num_);
                    sfence();
#endif
                }else{
                    if(allocate_index_ == total_num_){
                        log->allocate_object = nullptr;
                    }else{
                        // normal allocation
                        uint64_t* shift = reinterpret_cast<uint64_t*>(alloc_space + allocate_index_ * alloc_size);
                        *shift = allocate_index_;
#ifdef FLUSH                        
                        clwb(shift);
                        sfence();
#endif                    
                        log->allocate_object = reinterpret_cast<void*>(alloc_space + allocate_index_ * alloc_size + 8);
                        allocate_index_++;
#ifdef FLUSH
                        clwb(&allocate_index_);
                        sfence();
#endif
                    }
                }
            }

            void print_free_nodes(){
                auto node = free_nodes_;
                while(node != nullptr){
                    //printf("free addr is %p\n", node);
                    node = *((void**)node);
                }
            }   
        };

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
                            //std::cout << "Remove the chunk at " << i << std::endl;
                            return;
                        }
                    }
                }
            }
        };

    class PreAllocPool{
    public:
#ifndef DRAM_ALLOC        
        static PMEMoid p_pre_alloc_memory_;
        static PMEMoid p_new_chunk_;
#endif
        static memory_chunk* pre_alloc_memory_;// A linked list to manage all of the pre-allocated memory
#ifndef DRAM_ALLOC       
        static PMEMoid p_free_array_;
#endif
        static free_array* free_array_; // An array which manage the memory chunk which has free memory to use
        static size_t base_alloc_size_; // Every time allocated the same chunk size
        static allocate_log* log_;

        static void init(size_t base_alloc_size){ 
            // It should be 256-byte when allocating the small overflow chunk
#ifdef DRAM_ALLOC
            base_alloc_size_ = static_cast<size_t>(std::ceil((base_alloc_size + 8) / 64.) * 64); 
            free_array_ = reinterpret_cast<free_array*>(malloc(offsetof(free_array, free_array::free_memory_) + sizeof(memory_chunk*) * 1024));
            free_array_->initialize(1024);
            log_ = reinterpret_cast<allocate_log*>(malloc(sizeof(allocate_log)));
            pre_alloc_memory_ = reinterpret_cast<memory_chunk*>(malloc(CHUNK_SIZE));
            pre_alloc_memory_->initialize(nullptr, nullptr, (CHUNK_SIZE-48) / base_alloc_size_);
            std::cout << "base_alloc_size is " << base_alloc_size_ << std::endl;
            std::cout << "num_block in one memory chunk is " << (CHUNK_SIZE-48) / base_alloc_size_ << std::endl;
            // Only add it when there is node free
            free_array_->add(pre_alloc_memory_); 
            //printf("The addr of the new block is %p\n", pre_alloc_memory_);
#else           
            base_alloc_size_ = static_cast<size_t>(std::ceil((base_alloc_size + 8) / 64.) * 64); 
            BasePMPool::ZAllocate(&p_free_array_, offsetof(free_array, free_memory_) + sizeof(memory_chunk*) * 1024);
            free_array_ = reinterpret_cast<free_array*>(pmemobj_direct(p_free_array_));

            free_array_->initialize(1024);
            BasePMPool::ZAllocate(&p_pre_alloc_memory_, CHUNK_SIZE);
            pre_alloc_memory_ = reinterpret_cast<memory_chunk*>(pmemobj_direct(p_pre_alloc_memory_));
            pre_alloc_memory_->initialize(nullptr, nullptr, (CHUNK_SIZE-48) / base_alloc_size_);
            // Only add it when there is node free
            free_array_->add(pre_alloc_memory_); 
            p_new_chunk_ = OID_NULL;
            BasePMPool::ZAllocate((void**)&log_, sizeof(allocate_log));
#endif
        } 

        // Need the logic to allocate more memory
        // base_size is the size of the 
        static void* alloc_node(){
            auto free_num = free_array_->number_;
            if(free_num != 0){
                //directly use free node
                memory_chunk *free_chunk = free_array_->free_memory_[free_num - 1];
                free_chunk->allocate_one(base_alloc_size_, log_);
                if(free_chunk->is_full()){
                    free_array_->number_--;
                }
                return log_->allocate_object;
            }
            // Allocate more chunks
#ifdef DRAM_ALLOC            
            auto new_chunk = reinterpret_cast<memory_chunk*>(malloc(CHUNK_SIZE));
#else

#ifdef TEST_PRE
            memory_chunk* new_chunk;
            BasePMPool::PREAllocate((void**)&new_chunk, CHUNK_SIZE);
#else
            BasePMPool::ZAllocate(&p_new_chunk_, CHUNK_SIZE);
            auto new_chunk = reinterpret_cast<memory_chunk*>(pmemobj_direct(p_new_chunk_));
#endif

#endif
            //printf("allocate a new memory chunk %p\n", new_chunk);
            new_chunk->initialize(pre_alloc_memory_,nullptr,(CHUNK_SIZE-48)/base_alloc_size_);
            if(pre_alloc_memory_ != nullptr){
                pre_alloc_memory_->prev_ = new_chunk;
            }
            pre_alloc_memory_ = new_chunk;
            free_array_->add(new_chunk);
            new_chunk->allocate_one(base_alloc_size_, log_);
            if(new_chunk->is_full()){
                free_array_->number_--;
            }
            return log_->allocate_object;
        }

        static void free_node(void *node){
            uint64_t* size_shift = reinterpret_cast<uint64_t*>((char*)node - 8);
            char *cur_addr = reinterpret_cast<char*>(size_shift);
            memset(node, 0, base_alloc_size_ -8);
            memory_chunk* target_chunk = reinterpret_cast<memory_chunk*>(cur_addr - (*size_shift) * base_alloc_size_ - 48);
            //printf("free node at chunk %p with index %lld\n", target_chunk, *size_shift);
            //try to link
            void** ptr = reinterpret_cast<void**>(node);
            *ptr = target_chunk->free_nodes_;
#ifdef FLUSH
            clwb(ptr);
            sfence();
#endif
            target_chunk->free_nodes_ = node; //free nodes points to the real node
#ifdef FLUSH
            clwb(&target_chunk->free_nodes_);
            sfence();
#endif
            target_chunk->free_num_++;
#ifdef FLUSH
            clwb(&target_chunk->free_num_);
            sfence();
#endif
            //target_chunk->print_free_nodes();

// Reclaim the memory  space            
/*
            if(target_chunk->is_empty()){
                //Need to search in the free array to remove this chunk...
                //printf("remove chunk %p from free array\n", target_chunk);
                free_array_->remove(target_chunk);
#ifdef DRAM_ALLOC
                if(target_chunk->next_ != nullptr){
                    target_chunk->next_->prev_ = target_chunk->prev_;
                }

                if(target_chunk->prev_ != nullptr){
                    target_chunk->prev_->next_ = target_chunk->next_;
                }else{
                    std::cout << "update the first pointer\n" << std::endl;
                    pre_alloc_memory_ = target_chunk->next_;
                }

                free(target_chunk);
#else
                if(target_chunk->next_ != nullptr){
                    target_chunk->next_->prev_ = target_chunk->prev_;
                }

                if(target_chunk->prev_ != nullptr){
                    target_chunk->prev_->next_ = target_chunk->next_;
                }else{
                    //std::cout << "update the first pointer\n" << std::endl;
                    pre_alloc_memory_ = target_chunk->next_;
                }
                BasePMPool::Free(target_chunk);
#endif
                return;
            }
*/
            if((target_chunk->free_num_ == 1) && (target_chunk->allocate_index_ == target_chunk->total_num_)){
                //Need to expand the free array
                if(!free_array_->add(target_chunk)){
                    auto old_array = free_array_;
    #ifdef DRAM_ALLOC
                auto new_array = reinterpret_cast<free_array*>(malloc(offsetof(free_array, free_array::free_memory_) + sizeof(memory_chunk*) * old_array->total_num_ * 2));
                new_array->initialize(old_array->total_num_ * 2, old_array);
                free_array_ = new_array;
                free(old_array);
    #else 
                BasePMPool::ZAllocate((void**)&free_array_, offsetof(free_array, free_array::free_memory_) + sizeof(memory_chunk*) * old_array->total_num_ * 2);
                free_array_->initialize(old_array->total_num_ * 2, old_array);
                BasePMPool::Free(old_array);
    #endif
                free_array_->add(target_chunk);
                }
            }
        }

        static void display_info(){
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

#ifndef DRAM_ALLOC        
        PMEMoid PreAllocPool::p_pre_alloc_memory_ = OID_NULL;
        PMEMoid PreAllocPool::p_new_chunk_ = OID_NULL;
#endif
        memory_chunk* PreAllocPool::pre_alloc_memory_ = nullptr;// A linked list to manage all of the pre-allocated memory
#ifndef DRAM_ALLOC       
        PMEMoid PreAllocPool::p_free_array_ = OID_NULL;
#endif
        free_array* PreAllocPool::free_array_ = nullptr; // An array which manage the memory chunk which has free memory to use
        size_t PreAllocPool::base_alloc_size_ = 0; // Every time allocated the same chunk size
        allocate_log* PreAllocPool::log_ = nullptr;
}