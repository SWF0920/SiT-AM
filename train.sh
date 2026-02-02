export NCCL_P2P_DISABLE=1
export ENTITY="SiT_AM"
export PROJECT="SiT_AM_PROJECT"
export WANDB_MODE=online
export WANDB_KEY=""

torchrun --nnodes=1 --nproc_per_node=7 train.py \
  --model SiT-XL/2 \
  --wandb \
  --data-path ~/SiT/data/tiny-imagenet-200/train