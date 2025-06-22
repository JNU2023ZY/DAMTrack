#! /bin/bash

for i in {1..15}
do
    CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /data_F/zhouyong/DMAOT/dmaot/  DAM4SAM
done

vot pack --workspace /data_F/zhouyong/DMAOT/dmaot/  DAM4SAM
# for i in {1..15}
# do
#     CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /data2/cym/VOTS2023_Winner  swinb_dm_deaot
# done
