#!/bin/bash

#APEX evaluation
#./full.sh longitudes-200M.bin.data binary double 100000000 200000000 full 1 alex 1 0 > alex-longitudes-full.data
#./full.sh longitudes-200M.bin.data binary double 100000000 200000000 insert 1 alex 1 0 > alex-longitudes-insert.data
#./full.sh longitudes-200M.bin.data binary double 90000000 200000000 erase 1 alex 1 0 > alex-longitudes-erase.data

#./full.sh longlat-200M.bin.data binary double 100000000 200000000 full 1 alex 1 0 > alex-longlat-full.data
#./full.sh longlat-200M.bin.data binary double 100000000 200000000 insert 1 alex 1 0 > alex-longlat-insert.data
#./full.sh longlat-200M.bin.data binary double 90000000 200000000 erase 1 alex 1 0 > alex-longlat-erase.data

#./full.sh ycsb.data binary uint64 100000000 200000000 full 1 alex 1 0 > alex-ycsb-full.data
#./full.sh ycsb.data binary uint64 100000000 200000000 insert 1 alex 1 0 > alex-ycsb-insert.data
#./full.sh ycsb.data binary uint64 90000000 200000000 erase 1 alex 1 0 > alex-ycsb-erase.data

#./full.sh lognormal.data binary uint64 100000000 190000000 full 1 alex 1 0 > alex-lognormal-full.data
#./full.sh lognormal.data binary uint64 90000000 190000000 insert 1 alex 1 0 > alex-lognormal-insert.data
#./full.sh lognormal.data binary uint64 90000000 190000000 erase 1 alex 1 0 > alex-lognormal-erase.data

#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 full 1 alex 1 0 > alex-fb-full.data
#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 insert 1 alex 1 0 > alex-fb-insert.data
#./full.sh fb_200M_uint64 sosd uint64 90000000 200000000 erase 1 alex 1 0 > alex-fb-erase.data

#./full.sh tpce_trade_keys binary uint64 100000000 259200000 full 1 alex 1 0 > alex-tpce-full.data
#./full.sh tpce_trade_keys binary uint64 100000000 259200000 insert 1 alex 1 0 > alex-tpce-insert.data
#./full.sh tpce_trade_keys binary uint64 90000000 259200000 erase 1 alex 1 0 > alex-tpce-erase.data

#Bztree evaluation
#./full.sh longitudes-200M.bin.data binary double 100000000 200000000 full 0 bztree 0 0 > bztree-longitudes-full.data
#./full.sh longitudes-200M.bin.data binary double 100000000 200000000 insert 0 bztree 0 0 > bztree-longitudes-insert.data
#./full.sh longitudes-200M.bin.data binary double 90000000 200000000 erase 0 bztree 0 0 > bztree-longitudes-erase.data

#./full.sh longlat-200M.bin.data binary double 100000000 200000000 full 0 bztree 0 0 > bztree-longlat-full.data
#./full.sh longlat-200M.bin.data binary double 100000000 200000000 insert 0 bztree 0 0 > bztree-longlat-insert.data
#./full.sh longlat-200M.bin.data binary double 90000000 200000000 erase 0 bztree 0 0 > bztree-longlat-erase.data

#./full.sh ycsb.data binary uint64 100000000 200000000 full 0 bztree 0 0 > bztree-ycsb-full.data
#./full.sh ycsb.data binary uint64 100000000 200000000 insert 0 bztree 0 0 > bztree-ycsb-insert.data
#./full.sh ycsb.data binary uint64 90000000 200000000 erase 0 bztree 0 0 > bztree-ycsb-erase.data

./full.sh lognormal.data binary uint64 100000000 190000000 full 0 bztree 0 0 > bztree-lognormal-full.data
./full.sh lognormal.data binary uint64 90000000 190000000 insert 0 bztree 0 0 > bztree-lognormal-insert.data
./full.sh lognormal.data binary uint64 90000000 190000000 erase 0 bztree 0 0 > bztree-lognormal-erase.data

#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 full 0 bztree 0 0 > bztree-fb-full.data
#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 insert 0 bztree 0 0 > bztree-fb-insert.data
#./full.sh fb_200M_uint64 sosd uint64 90000000 200000000 erase 0 bztree 0 0 > bztree-fb-erase.data

./full.sh tpce_trade_keys binary uint64 100000000 259200000 full 0 bztree 0 0 > bztree-tpce-full.data
./full.sh tpce_trade_keys binary uint64 100000000 259200000 insert 0 bztree 0 0 > bztree-tpce-insert.data
./full.sh tpce_trade_keys binary uint64 90000000 259200000 erase 0 bztree 0 0 > bztree-tpce-erase.data

# fastfair evaluation
#./full.sh longitudes-200M.bin.data binary double 100000000 200000000 full 0 fastfair 0 0 > fastfair-longitudes-full.data
#./full.sh longitudes-200M.bin.data binary double 100000000 200000000 insert 0 fastfair 0 0 > fastfair-longitudes-insert.data
#./full.sh longitudes-200M.bin.data binary double 90000000 200000000 erase 0 fastfair 0 0 > fastfair-longitudes-erase.data

#./full.sh longlat-200M.bin.data binary double 100000000 200000000 full 0 fastfair 0 0 > fastfair-longlat-full.data
#./full.sh longlat-200M.bin.data binary double 100000000 200000000 insert 0 fastfair 0 0 > fastfair-longlat-insert.data
#./full.sh longlat-200M.bin.data binary double 90000000 200000000 erase 0 fastfair 0 0 > fastfair-longlat-erase.data

#./full.sh ycsb.data binary uint64 100000000 200000000 full 0 fastfair 0 0 > fastfair-ycsb-full.data
#./full.sh ycsb.data binary uint64 100000000 200000000 insert 0 fastfair 0 0 > fastfair-ycsb-insert.data
#./full.sh ycsb.data binary uint64 90000000 200000000 erase 0 fastfair 0 0 > fastfair-ycsb-erase.data

#./full.sh lognormal.data binary uint64 100000000 190000000 full 0 fastfair 0 0 > fastfair-lognormal-full.data
#./full.sh lognormal.data binary uint64 90000000 190000000 insert 0 fastfair 0 0 > fastfair-lognormal-insert.data
#./full.sh lognormal.data binary uint64 90000000 190000000 erase 0 fastfair 0 0 > fastfair-lognormal-erase.data

#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 full 0 fastfair 0 0 > fastfair-fb-full.data
#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 insert 0 fastfair 0 0 > fastfair-fb-insert.data
#./full.sh fb_200M_uint64 sosd uint64 90000000 200000000 erase 0 fastfair 0 0 > fastfair-fb-erase.data

./full.sh tpce_trade_keys binary uint64 100000000 259200000 full 0 fastfair 0 0 > fastfair-tpce-full.data
./full.sh tpce_trade_keys binary uint64 100000000 259200000 insert 0 fastfair 0 0 > fastfair-tpce-insert.data
./full.sh tpce_trade_keys binary uint64 90000000 259200000 erase 0 fastfair 0 0 > fastfair-tpce-erase.data

# lbtree uint64_t evaluation
./full.sh ycsb.data binary uint64 100000000 200000000 full 0 lbtree 1 0 > lbtree-ycsb-full.data
./full.sh ycsb.data binary uint64 100000000 200000000 insert 0 lbtree 1 0 > lbtree-ycsb-insert.data
./full.sh ycsb.data binary uint64 90000000 200000000 erase 0 lbtree 1 0 > lbtree-ycsb-erase.data

./full.sh lognormal.data binary uint64 100000000 190000000 full 0 lbtree 1 0 > lbtree-lognormal-full.data
./full.sh lognormal.data binary uint64 90000000 190000000 insert 0 lbtree 1 0 > lbtree-lognormal-insert.data
./full.sh lognormal.data binary uint64 90000000 190000000 erase 0 lbtree 1 0 > lbtree-lognormal-erase.data

#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 full 0 lbtree 1 0 > lbtree-fb-full.data
#./full.sh fb_200M_uint64 sosd uint64 100000000 200000000 insert 0 lbtree 1 0 > lbtree-fb-insert.data
#./full.sh fb_200M_uint64 sosd uint64 90000000 200000000 erase 0 lbtree 1 0 > lbtree-fb-erase.data

./full.sh tpce_trade_keys binary uint64 100000000 259200000 full 0 lbtree 1 0 > lbtree-tpce-full.data
./full.sh tpce_trade_keys binary uint64 100000000 259200000 insert 0 lbtree 1 0 > lbtree-tpce-insert.data
./full.sh tpce_trade_keys binary uint64 90000000 259200000 erase 0 lbtree 1 0 > lbtree-tpce-erase.data


# Mixed workload, only select two representative data sets to do
./full.sh longitudes-200M.bin.data binary double 100000000 200000000 mixed 1 alex 1 0.2 > alex-longitudes-mixed02.data
./full.sh longitudes-200M.bin.data binary double 100000000 200000000 mixed 1 alex 1 0.5 > alex-longitudes-mixed05.data

./full.sh longlat-200M.bin.data binary double 100000000 200000000 mixed 1 alex 1 0.2 > alex-longlat-mixed02.data
./full.sh longlat-200M.bin.data binary double 100000000 200000000 mixed 1 alex 1 0.5 > alex-longlat-mixed05.data

./full.sh longitudes-200M.bin.data binary double 100000000 200000000 mixed 0 bztree 0 0.2 > bztree-longitudes-mixed02.data
./full.sh longitudes-200M.bin.data binary double 100000000 200000000 mixed 0 bztree 0 0.5 > bztree-longitudes-mixed05.data

./full.sh longlat-200M.bin.data binary double 100000000 200000000 mixed 0 bztree 0 0.2 > bztree-longlat-mixed02.data
./full.sh longlat-200M.bin.data binary double 100000000 200000000 mixed 0 bztree 0 0.5 > bztree-longlat-mixed05.data

./full.sh longitudes-200M.bin.data binary double 100000000 200000000 mixed 0 fastfair 0 0.2 > fastfair-longitudes-mixed02.data
./full.sh longitudes-200M.bin.data binary double 100000000 200000000 mixed 0 fastfair 0 0.5 > fastfair-longitudes-mixed05.data

./full.sh longlat-200M.bin.data binary double 100000000 200000000 mixed 0 fastfair 0 0.2 > fastfair-longlat-mixed02.data
./full.sh longlat-200M.bin.data binary double 100000000 200000000 mixed 0 fastfair 0 0.5 > fastfair-longlat-mixed05.data