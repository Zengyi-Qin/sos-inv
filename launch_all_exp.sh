#!/bin/bash

run_parallel() {
    while [ "$(jobs -r | wc -l)" -ge 4 ]; do
        sleep 10
    done
    eval "$*" &  # Use eval on the entire command string
}

# focused wave
run_parallel CUDA_VISIBLE_DEVICES=0 python train.py --input_type focused_wave \
    --num_transmits 1 \
    --ckp ./outputs/fw_trn_1/checkpoint \
    --vis ./outputs/fw_trn_1/vis

run_parallel CUDA_VISIBLE_DEVICES=1 python train.py --input_type focused_wave \
    --num_transmits 3 \
    --ckp ./outputs/fw_trn_3/checkpoint \
    --vis ./outputs/fw_trn_3/vis

run_parallel CUDA_VISIBLE_DEVICES=2 python train.py --input_type focused_wave \
    --num_transmits 5 \
    --ckp ./outputs/fw_trn_5/checkpoint \
    --vis ./outputs/fw_trn_5/vis

run_parallel CUDA_VISIBLE_DEVICES=3 python train.py --input_type focused_wave \
    --num_transmits 7 \
    --ckp ./outputs/fw_trn_7/checkpoint \
    --vis ./outputs/fw_trn_7/vis

run_parallel CUDA_VISIBLE_DEVICES=0 python train.py --input_type focused_wave \
    --num_transmits 9 \
    --ckp ./outputs/fw_trn_9/checkpoint \
    --vis ./outputs/fw_trn_9/vis

run_parallel CUDA_VISIBLE_DEVICES=1 python train.py --input_type focused_wave \
    --num_transmits 11 \
    --ckp ./outputs/fw_trn_11/checkpoint \
    --vis ./outputs/fw_trn_11/vis

# plane wave
run_parallel CUDA_VISIBLE_DEVICES=2 python train.py --input_type plane_wave \
    --num_transmits 1 \
    --ckp ./outputs/pw_trn_1/checkpoint \
    --vis ./outputs/pw_trn_1/vis

run_parallel CUDA_VISIBLE_DEVICES=3 python train.py --input_type plane_wave \
    --num_transmits 3 \
    --ckp ./outputs/pw_trn_3/checkpoint \
    --vis ./outputs/pw_trn_3/vis

run_parallel CUDA_VISIBLE_DEVICES=0 python train.py --input_type plane_wave \
    --num_transmits 5 \
    --ckp ./outputs/pw_trn_5/checkpoint \
    --vis ./outputs/pw_trn_5/vis

run_parallel CUDA_VISIBLE_DEVICES=1 python train.py --input_type plane_wave \
    --num_transmits 7 \
    --ckp ./outputs/pw_trn_7/checkpoint \
    --vis ./outputs/pw_trn_7/vis

run_parallel CUDA_VISIBLE_DEVICES=2 python train.py --input_type plane_wave \
    --num_transmits 9 \
    --ckp ./outputs/pw_trn_9/checkpoint \
    --vis ./outputs/pw_trn_9/vis

run_parallel CUDA_VISIBLE_DEVICES=3 python train.py --input_type plane_wave \
    --num_transmits 11 \
    --ckp ./outputs/pw_trn_11/checkpoint \
    --vis ./outputs/pw_trn_11/vis

# scan line
run_parallel CUDA_VISIBLE_DEVICES=0 python train.py --input_type scan_line \
    --ckp ./outputs/sl/checkpoint \
    --vis ./outputs/sl/vis

# Wait for all background processes to finish
wait

echo "All tasks completed."
