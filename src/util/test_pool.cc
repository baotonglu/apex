#include "preallocpool.h"
#include <utility>
#include <cstdint>

 class OverflowBlock{
    public: 
    typedef std::pair<uint64_t, uint64_t> V;
    uint64_t bitmap_; /*whether the bitmap should be persistent?*/
    OverflowBlock *next; 
    V data_slots_[14]; 
 };


int main(){
    my_alloc::PreAllocPool::init(124);
    // Test 1
    void *nodes[4];
    for(int i = 0; i < 4; ++i){
        nodes[i] = my_alloc::PreAllocPool::alloc_node();
        printf("New node addr is %p\n", nodes[i]);
    }
    
    printf("Finish the allocation\n");

    my_alloc::PreAllocPool::display_info();

    printf("start deallocate the nodes\n");
    for(int i = 0; i < 2; ++i){
        my_alloc::PreAllocPool::free_node(nodes[i]);
    }

    printf("start allocate four new nodes\n");
    for(int i = 0; i < 4; ++i){
        nodes[i] = my_alloc::PreAllocPool::alloc_node();
        printf("New node addr is %p\n", nodes[i]);
    }

    my_alloc::PreAllocPool::display_info();

    std::cout << "The size of overflow block is "<< sizeof(OverflowBlock) << std::endl;

    return 0;
}