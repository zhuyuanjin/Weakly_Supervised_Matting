#!/bin/bash
#SBATCH -o log_baseline
#SBATCH --mem=60G
#SBATCH -J Baseline               # 作业名为 test
#SBATCH -p geforce                # 作业提交的分区为 cpu
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=2       # 单节点启动的进程数为 1
#SBATCH -t 24:00:00               # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡
#SBATCH --mem-per-gpu=30G

source activate pytorch0.4
python train.py 
