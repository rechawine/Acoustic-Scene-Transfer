#!/bin/bash
#SBATCH -J AST
#SBATCH -p mt_a40
#SBATCH -w gpu251
#SBATCH --qos=high
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH -o train.log


nvidia-smi

export HOME=/ssd12/other/ziqiliang/data/RapBank/data_pipeline/seperation/liangzq02
export PATH=/ssd1/shared/tool/gcc-5.4.0/bin:$PATH
export LD_LIBRARY_PATH=/ssd1/shared/tool/gcc-5.4.0/lib64
export PATH=/ssd1/slurm/bin:$PATH
export PATH=/ssd1/shared/local/cuda-10.2/bin${PATH:+:${PATH}}
export CUDA_HOME=/ssd1/shared/local/cuda-10.2
export LD_LIBRARY_PATH=/ssd1/shared/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 接收GPU ID作为参数，默认使用0,1,2,3
GPU_IDS=1,2,3,4
# 设置要使用的GPU ID
export CUDA_VISIBLE_DEVICES=$GPU_IDS
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# 计算GPU数量
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# 使用accelerate启动分布式训练
accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    train.py --config configs/VoiceLDM-M.yaml --output_dir outputs_slurm