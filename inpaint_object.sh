#!/bin/bash         

curr_dir=$(pwd)

frames_dir=$1
tmp_dir=$2

if [ -d $tmp_dir ]; then
    rm -rf $tmp_dir
fi

mkdir -p $tmp_dir

tmp_masks_dir=$tmp_dir/masks
tmp_video_dir=$tmp_dir/video_out

python dam4sam/run_bbox_example.py --dir $frames_dir --output_dir $tmp_masks_dir

python proPainter/inference_propainter.py --video $frames_dir --mask $tmp_masks_dir --output $tmp_video_dir --resize_ratio 0.5
echo "Done."
