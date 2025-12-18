export NCCL_P2P_DISABLE=1
export ENTITY="SiT_AM"
export PROJECT="SiT_AM_PROJECT"
export WANDB_MODE=online
export WANDB_KEY=""

torchrun --nnodes=1 --nproc_per_node=7 finetune/finetune_am.py \
    --ckpt ~/SiT/pretrained_models/SiT-XL-2-256x256.pt \
    --method full \
    --reward quadratic \
    --reward-scale 100.0 \
    --lr 5e-4 \
    --batch-size 16 \
    --grad-clip 10.0 \
    --num-sampling-steps 20 \
    --num-iterations 5000 \
    --log-every 1 \
    --wandb