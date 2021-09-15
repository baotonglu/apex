#pragma once

#include <iostream>
#include <cstddef>
#include <climits>
#include <cstdlib>
#include <new>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "libpmem.h"
#include "libpmemobj.h"
#include <garbage_list.h>

#include "utils.h"
 
// In this class, I will write a custom template allocator
// Specifically, it allocates persistent memory using PMDK interface
// Moreover, need to use static member to make all allocatoion in a single memory pool

static const char* layout_name = "template_pool";
static const uint64_t pool_addr = 0x5f0000000000;
static const char* pool_name = "/mnt/pmem0/baotong/template.data";
static const uint64_t pool_size = 40UL * 1024*1024*1024;

namespace my_alloc{

typedef void (*DestroyCallback)(void* callback_context, void* object);

	template <class T1, class T2>
	inline void _construct(T1* p, const T2& value){new (p) T1(value);}

	template <class T>
	inline void _destroy(T* ptr){ ptr->~T();}

    //Implement a base class that has the memory pool
class BasePMPool{
public:
    static PMEMobjpool *pm_pool_;
    static int allocator_num;
    static const uint64_t kAllTables = 4UL * 1024 * 1024 * 1024;    
    static PMEMoid p_all_tables;
    static char *all_tables;
    static uint64_t all_allocated;
    static uint64_t all_deallocated;
    static BasePMPool* instance_;

    EpochManager epoch_manager_{};
    GarbageList garbage_list_{};

    static bool Initialize(const char* pool_name, size_t pool_size){
        //if(pm_pool_ == nullptr){
        bool recover = false;
        if (!FileExists(pool_name)) {
            LOG("creating a new pool");
            pm_pool_ = pmemobj_create_addr(pool_name, layout_name, pool_size,
                                            CREATE_MODE_RW, (void*)pool_addr);
            if (pm_pool_ == nullptr) {
                LOG_FATAL("failed to create a pool;");
            }
            std::cout << "pool opened at: " << std::hex << pm_pool_ << std::dec << std::endl;
        }else{
            LOG("opening an existing pool, and trying to map to same address");
            /* Need to open an existing persistent pool */
            recover = true;
            pm_pool_ = pmemobj_open_addr(pool_name, layout_name, (void*)pool_addr);
            if (pm_pool_ == nullptr) {
                LOG_FATAL("failed to open the pool");
            }
            std::cout << "pool opened at: " << std::hex << pm_pool_
                << std::dec << std::endl;
        }

        instance_ = new BasePMPool();
        instance_->epoch_manager_.Initialize();
        instance_->garbage_list_.Initialize(&instance_->epoch_manager_, instance_->pm_pool_, 1024 * 8);

        IncreaseAllocatorNum();
        return recover;
    }

    static void IncreaseAllocatorNum(){
        allocator_num++;
    }

    static void DecreaseAllocatorNum(){
        allocator_num--;
    }

    static void ClosePool(){
        if(pm_pool_ != nullptr){
            pmemobj_close(pm_pool_);
        }
    }

    static void* GetRoot(size_t size) {
        return pmemobj_direct(pmemobj_root(pm_pool_, size));
    }

    static void AlignAllocate(void** ptr, size_t size){
        PMEMoid tmp_ptr;
        auto ret = pmemobj_alloc(pm_pool_, &tmp_ptr, size + 64, TOID_TYPE_NUM(char), NULL, NULL);
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << size << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 1");
        }
        uint64_t ptr_value = (uint64_t)(pmemobj_direct(tmp_ptr)) + 48;
        *ptr = (void*)(ptr_value);
    }

    static void AlignZAllocate(void** ptr, size_t size){
        PMEMoid tmp_ptr;
        auto ret = pmemobj_zalloc(pm_pool_, &tmp_ptr, size + 64, TOID_TYPE_NUM(char));
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << size << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 1");
        }
        uint64_t ptr_value = (uint64_t)(pmemobj_direct(tmp_ptr)) + 48;
        *ptr = (void*)(ptr_value);
    }

    static void Free(void* p){ 
        auto ptr = pmemobj_oid(p);
        pmemobj_free(&ptr);
    }

    //Need to address this
    static void Allocate(void** ptr, size_t size){
        PMEMoid tmp_ptr;
        auto ret = pmemobj_alloc(pm_pool_, &tmp_ptr, size, TOID_TYPE_NUM(char), NULL, NULL);
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << size << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 1");
        }
        *ptr = pmemobj_direct(tmp_ptr);
    }

    static void ZAllocate(void** ptr, size_t size){
        PMEMoid tmp_ptr;
        auto ret = pmemobj_zalloc(pm_pool_, &tmp_ptr, size, TOID_TYPE_NUM(char));
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << size << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 1");
        }
        *ptr = pmemobj_direct(tmp_ptr);
    }

    static void Allocate(PMEMoid *ptr, size_t size){
        auto ret = pmemobj_alloc(pm_pool_, ptr, size, TOID_TYPE_NUM(char), NULL, NULL);
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << size << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 1");
        }
    }

    static void Allocate(PMEMoid* pm_ptr, uint32_t alignment, size_t size,
                       int (*alloc_constr)(PMEMobjpool* pool, void* ptr,
                                           void* arg),
                       void* arg) {
    auto ret = pmemobj_alloc(pm_pool_, pm_ptr, size,
                             TOID_TYPE_NUM(char), alloc_constr, arg);
        if (ret) {
        LOG_FATAL("Allocate Initialize: Allocation Error in PMEMoid");
        }
    }

    static void ZAllocate(PMEMoid *ptr, size_t size){
        auto ret = pmemobj_zalloc(pm_pool_, ptr, size, TOID_TYPE_NUM(char));
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << size << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 1");
        }
    }

    static void AlignFree(void* p){ 
        uint64_t ptr_value = (uint64_t)(p) - 48;
        void *new_p = reinterpret_cast<void*>(ptr_value);
        auto ptr = pmemobj_oid(new_p);
        pmemobj_free(&ptr);
    }

    static void Persist(void* p, size_t size){
        pmemobj_persist(pm_pool_, p, size);
    }

    static void DefaultPMCallback(void* callback_context, void* ptr) {
        auto oid_ptr = pmemobj_oid(ptr);
        TOID(char) ptr_cpy;
        TOID_ASSIGN(ptr_cpy, oid_ptr);
        POBJ_FREE(&ptr_cpy);
    }

    static void DefaultDRAMCallback(void* callback_context, void* ptr) {
        free(ptr);
    }

    static void SafeFree(void* ptr, DestroyCallback callback = DefaultPMCallback,
                    void* context = nullptr) {
        instance_->garbage_list_.Push(ptr, callback, context);
    }

    static void SafeFree(GarbageList::Item* item, void* ptr,
                    DestroyCallback callback = DefaultPMCallback,
                    void* context = nullptr) {
        item->SetValue(ptr, instance_->epoch_manager_.GetCurrentEpoch(), callback,
                    context);
    }

    static EpochGuard AquireEpochGuard() {
        return EpochGuard{&instance_->epoch_manager_};
    }

    static void Protect() { instance_->epoch_manager_.Protect(); }

    static void Unprotect() { instance_->epoch_manager_.Unprotect(); }

    static GarbageList::Item* ReserveItem() {
        return instance_->garbage_list_.ReserveItem();
    }

    static void ResetItem(GarbageList::Item* mem) {
        instance_->garbage_list_.ResetItem(mem);
    }

    static void EpochRecovery() {
        instance_->garbage_list_.Recovery(&instance_->epoch_manager_,
                                        instance_->pm_pool_);
    }
};

	
template <class T>
class allocator : BasePMPool{
public: 
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;

	//constructor fucn
    allocator(){
        std::cout << "Intial allocator: " << allocator_num << std::endl;
        ADD(&allocator_num, 1);
        if(allocator_num == 1){
            BasePMPool::Initialize(pool_name, pool_size);
        }
    }

    allocator(const allocator<T>& c){
        ADD(&allocator_num, 1);
        if(allocator_num == 1){
            BasePMPool::Initialize(pool_name, pool_size);
        }
    }

    ~allocator(){
        if(allocator_num == 0){
            //delete PM pool
            ClosePool();
        }
    }

	template <class U>
	allocator(const allocator<U>& c){
        if(allocator_num == 1){
            BasePMPool::Initialize(pool_name, pool_size);
        }
    }

	//rebind allocator of type U
	template <class U>
	struct rebind {typedef allocator<U> other;};

	//pointer allocate(size_type n, const void* hint=0){
    pointer allocate(size_type n){
        //FIXME: non-safe memory allocation
        PMEMoid tmp_ptr;
        auto ret = pmemobj_alloc(pm_pool_, &tmp_ptr, (size_t)(n * sizeof(T)), TOID_TYPE_NUM(char), NULL, NULL);
        if (ret) {
          std::cout << "Fail logging: " << ret << "; Size = " << n *sizeof(T) << std::endl;
          LOG_FATAL("Allocate: Allocation Error in PMEMoid 2");
        }
        pointer tmp = (pointer)pmemobj_direct(tmp_ptr);
        return tmp;
	}

	void deallocate(pointer p, size_type n){ 
        auto ptr = pmemobj_oid(p);
        pmemobj_free(&ptr);
    }
	void construct(pointer p, const T& value){ _construct(p, value);}
	void destroy(pointer p){_destroy(p);}

	pointer address(reference x){return (pointer)&x;}
	const_pointer const_address(const_reference x){ return (const_pointer)&x;}

	size_type max_size() const {return size_type(UINT_MAX / sizeof(T));}
};

PMEMobjpool* BasePMPool::pm_pool_ = nullptr;
int BasePMPool::allocator_num = 0;
PMEMoid BasePMPool::p_all_tables = OID_NULL;
char* BasePMPool::all_tables = nullptr;
uint64_t BasePMPool::all_allocated = 0;
uint64_t BasePMPool::all_deallocated = 0;
BasePMPool* BasePMPool::instance_ = nullptr;
}
