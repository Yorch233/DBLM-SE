#!/bin/bash
#SBATCH --job-name=token
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224   
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00   
#SBATCH --mem=1024G 
#SBATCH --exclusive

export LOGLEVEL=INFO

export NCCL_DEBUG=INFO

# 定义每个节点的自定义参数
PARAMS=("split_1.txt" "split_2.txt" "split_3.txt" "split_4.txt" "split_5.txt")

scontrol show hostnames "$SLURM_JOB_NODELIST" > node_list.txt
mapfile -t NODES < node_list.txt
rm node_list.txt 
# 检查节点数量是否匹配参数数量
if [ "${#NODES[@]}" -ne "${#PARAMS[@]}" ]; then
    echo "节点数量与参数数量不匹配！"
    exit 1
fi

# 在每个节点上启动 torchrun
for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    PARAM=${PARAMS[$i]}
    
    echo "在节点 $NODE 上启动 torchrun，参数: $PARAM"
    
    srun --nodes=1 --ntasks=1 --cpus-per-task=224 --gres=gpu:8 --exclusive -w "$NODE" \
        torchrun --nnodes=1 \
                 --nproc_per_node=8 \
                 inference_save_code.py --flist_file="$PARAM"&
done

# 等待所有后台任务完成
wait

echo "所有 torchrun 任务已完成。"
