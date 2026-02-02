export NCCL_P2P_DISABLE=1
export ENTITY="SiT_AM"
export PROJECT="SiT_AM_PROJECT"
export WANDB_MODE=online
export WANDB_KEY="1014c7509fec4b824247"

torchrun --nnodes=1 --nproc_per_node=7 --master_port=29501 finetune_am.py \
    --ckpt ~/SiT/pretrained_models/SiT-XL-2-256x256.pt \
    --method original \
    --reward imagereward \
    --reward-scale 20000.0 \
    --cfg-scale 1.5 \
    --lr 6e-5 \
    --batch-size 16 \
    --grad-clip 10.0 \
    --num-sampling-steps 50 \
    --num-iterations 40 \
    --log-every 1 \
    --wandb

torchrun --nnodes=1 --nproc_per_node=7 --master_port=29502 finetune_am.py \
    --ckpt ~/SiT/pretrained_models/SiT-XL-2-256x256.pt \
    --method stochastic \
    --reward imagereward \
    --reward-scale 20000.0 \
    --cfg-scale 1.5 \
    --lr 6e-5 \
    --batch-size 16 \
    --grad-clip 10.0 \
    --num-sampling-steps 50 \
    --num-iterations 40 \
    --log-every 1 \
    --wandb