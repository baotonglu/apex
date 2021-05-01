#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>

#include "libpmem.h"
#include "libpmemobj.h"
#include "../util/allocator.h"

// LOG descriptor, specific to APEX, the idea is somewhat similar to micro-log used in FPTree

// Redo-Undo log, could be used in data node resize and overflow stash allocation
// After new_node_ is set, we need redo the step
// Need cacheline alignment
struct __attribute__((aligned(64))) ResizeLog {
    PMEMoid cur_node_ = OID_NULL; // Use this as the indicator
    PMEMoid new_node_ = OID_NULL; // Only start redo when new_node_ is allocated
    int progress_ = 0; // 0 means nothing ocurrs; 1 means resize is in initial phase (undo); 2 means the data node is well prepared, redo is better
    char key_[8]; // maximum key size is 8B
    char dummy[20]; // For cacheline alignment

    void clear_log(){
        progress_ = 0;
        my_alloc::BasePMPool::Persist(&progress_, sizeof(progress_));
        memset(this, 0, sizeof(ResizeLog));
        my_alloc::BasePMPool::Persist(this, sizeof(ResizeLog));
    }
};

//Redo-Undo-log
//Rollback function needs to be written
struct RootExpandLog{
    void clear_log(){
        root_expand_in_progress_ = 0;
        my_alloc::BasePMPool::Persist(&root_expand_in_progress_, sizeof(root_expand_in_progress_));
        old_root_node_ = OID_NULL;
        old_children_ = OID_NULL;
        new_children_ = OID_NULL;
        new_root_node_ = OID_NULL;
        outermost_node_ = OID_NULL;
        a_ = 0;
        b_ = 0;
        num_keys_at_last_domain_resize_ = 0;
        num_children_ = 0;
        new_nodes_end_ = 0;
        new_nodes_start_ = 0;
        new_domain_min_ = 0;
        new_domain_max_ = 0;
        root_expand_in_progress_ = 0;
        my_alloc::BasePMPool::Persist(this, sizeof(RootExpandLog));
    }

    PMEMoid old_root_node_ = OID_NULL; // The root node that needs expansion 
    PMEMoid old_children_ = OID_NULL;
    PMEMoid new_children_ = OID_NULL;

    PMEMoid new_root_node_ = OID_NULL;
    PMEMoid outermost_node_ = OID_NULL;
    // Model in root node
    double a_ = 0;
    double b_ = 0;

    // parameter
    int num_keys_at_last_domain_resize_ = 0;
    int num_children_ = 0;
    int new_nodes_end_ = 0;
    int new_nodes_start_ = 0;

    //One cacheline size
    double new_domain_min_ = 0;
    double new_domain_max_ = 0;

    // -1 means exapnd left start; 1 means expand right end; 0 means no root exapnd; 
    // -2 means creat new root node left; 2 means create new root node right
    int root_expand_in_progress_ = 0; 
    bool start_execute = false; // 0 means haven't in node_remove_
};
    
// Log becomes redo after new node is allocated
struct NewModelResizeLog{
    PMEMoid cur_node_ = OID_NULL;
    PMEMoid new_node_ = OID_NULL;
};

// SMO for node split
// Redo after the node has allocated the new data nodes since the node links are already updated after this
struct __attribute__((aligned(64))) SplitSidewayDownwardsLog{
    void clear_log(){
        progress_ = 0;
        my_alloc::BasePMPool::Persist(&progress_, sizeof(progress_));
        if(!OID_IS_NULL(child_nodes_)){
            pmemobj_free(&child_nodes_);
        }
        memset(this, 0, sizeof(SplitSidewayDownwardsLog));
        my_alloc::BasePMPool::Persist(this, sizeof(SplitSidewayDownwardsLog));
    }

    // Used for store new nodes
    PMEMoid reserved_child_[2]; // For 2 children
    PMEMoid child_nodes_ = OID_NULL; // For > 2 children; should also have the storage for tree level (also local depth)
    // Used when split downards
    PMEMoid downward_node = OID_NULL; 
   
    // Used when model resizing
    NewModelResizeLog model_resize_log_;

    // Basic info of this log
    PMEMoid cur_node_ = OID_NULL; // point to old node
    int split_num_ = 0; // Number of children, only updated when #children > 2
    // 0 means this process is not in a SMO operation
    // > 0 means the SMO is ocurring
    // 1 means normal state of of SMO
    // 2 means split sideways downwards is ocurring
    // 3 means model node expand occurs
    uint16_t progress_ = 0;
    uint16_t fanout_tree_depth = 0;
    char key_[8]; // inserting key, using key to find the parent    
};

// The log for overflow stash block allcocation
struct OverflowDesc{
    PMEMoid new_node_ = OID_NULL;
    void *cur_node_; // a pointer data node this is allocating the overflow stash
    uint64_t dummy;
};

// This class has all log that could be used by one thread
class MyLog{
public:
    ResizeLog resize_log_; //Data Node Resize, only used during resizing
    SplitSidewayDownwardsLog smo_log_; // More complex SMO operation
    OverflowDesc overflow_desc_; // The descriptor for safe allocation of overflow block
    uint64_t in_use; // 0 means that this node is not in-use; registe using thread ID
    uint64_t dummy[3];
};

// Log array is like a persistent log pool
// MyLog is asigned to threads which need to operate on APEX
// log_array_ needs cacheline alignment
class LogArray{
public:
    MyLog log_array_[1024];// At most, 1024 threads use this log array

    LogArray(){
        clear_log();
    }

    void clear_log(){
        memset(log_array_, 0, sizeof(MyLog) * 1024);
        my_alloc::BasePMPool::Persist(log_array_, sizeof(MyLog) * 1024);
    }

    int log_size(){
        return 1024;
    }

    // Find the existing memory pool or assign a new memory pool to this thread
    MyLog* assign_log(uint64_t* assign_index){
        uint64_t current_thread_id = pthread_self();
        // first scan to see whether I already has a memory pool registerd
        for(int i = 0; i < 1024; ++i){
            if(log_array_[i].in_use == current_thread_id){
                *assign_index = i;
                return &(log_array_[i]);
            }
        }

        // regitster the thread id in this pool
        while(true){
            for(int i = 0; i < 1024; ++i){
                uint64_t old_value = 0;
                uint64_t new_value = current_thread_id;
                if(CAS(&log_array_[i].in_use, &old_value, new_value)){
                    *assign_index = i;
                    return &(log_array_[i]);
                }
            }
        }
    }

   void return_log(uint64_t assign_index){
        uint64_t current_thread_id = pthread_self();
        if(log_array_[assign_index].in_use == current_thread_id){
            log_array_[assign_index].in_use = 0;
        }
   }
};