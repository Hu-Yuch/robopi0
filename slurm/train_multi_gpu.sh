#!/bin/bash

#SBATCH --job-name=pg-vla
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G

export WANDB__SERVICE_WAIT=300
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Hugging Face download settings
export HF_HOME=$HOME/.cache/huggingface  # 设置缓存目录
export HF_ENDPOINT=https://hf-mirror.com  # 主镜像
export HF_DATASETS_OFFLINE=0
export HF_EVALUATE_OFFLINE=0
export HF_MODULES_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DOWNLOAD_TIMEOUT=600  # 增加下载超时时间到10分钟
# 备用镜像设置
export TRANSFORMERS_OFFLINE=0
export HF_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models
export WANDB_BASE_URL=https://api.bandw.top

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

# run script with selected configuration using torchrun
HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run.py \
  --config-name=robobrain_calvin_2 \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  flow_sampling=beta \
  use_torch_compile=True \
  use_bf16=True \
  use_amp=True
