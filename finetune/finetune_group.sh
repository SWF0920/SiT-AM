export NCCL_P2P_DISABLE=1
export ENTITY="SiT_AM"
export PROJECT="SiT_AM_PROJECT"
export WANDB_MODE=online
export WANDB_KEY=""

LR_LIST=("4e-4" "2e-4" "1e-4")
SCALE_LIST=("20.0" "50.0" "100.0")
CLIP_LIST=("20.0" "100.0")

BATCH_SIZE=16
NUM_STEPS=20
NUM_ITERS=80

# optional: group runs together in wandb
export WANDB_RUN_GROUP="quadratic_sweep_$(date +%Y%m%d_%H%M%S)"

# ====== sweep loop ======
for lr in "${LR_LIST[@]}"; do
  for scale in "${SCALE_LIST[@]}"; do
    for clip in "${CLIP_LIST[@]}"; do

      RUN_NAME="quad_lr${lr}_scale${scale}_clip${clip}"
      echo "====================================================="
      echo "Starting run: ${RUN_NAME}"
      echo "  lr=${lr}, reward-scale=${scale}, grad-clip=${clip}"
      echo "====================================================="

      # per-run name for wandb (if your finetune script respects it)
      WANDB_NAME="${RUN_NAME}" torchrun --nnodes=1 --nproc_per_node=7 finetune/finetune_am.py \
        --ckpt ~/SiT/pretrained_models/SiT-XL-2-256x256.pt \
        --method full \
        --reward quadratic \
        --reward-scale "${scale}" \
        --lr "${lr}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-clip "${clip}" \
        --num-sampling-steps "${NUM_STEPS}" \
        --num-iterations "${NUM_ITERS}" \
        --log-every 1 \
        --wandb

      # if a run crashes, don't kill the whole sweep
      if [ $? -ne 0 ]; then
        echo "Run ${RUN_NAME} failed (non-zero exit). Continuing to next..."
      fi

      echo "Finished run: ${RUN_NAME}"
      echo
    done
  done
done

echo "All sweeps finished."
