#!/bin/bash
# Script for linear probing on TPU
AGGRE_TYPE="cls_token" # cls_token or global_pool
TPU_NAME=mae  # !!! change to the TPU name you created
MAE_PATH=/checkpoint/qianlim/workspace/mae_tpu  # where the repo is cloned above
IMAGENET_DIR=/checkpoint/imagenet-1k/  # where ImageNet-1k is downloaded above
mask_ratio=$1
patch_size=$2
BATCH_SIZE_PER_TPU=128  # 16384 (total batch size) // 128 (tpu cores)
SAVE_DIR="/checkpoint/qialim/mae_save/vitb_800ep_m${mask_ratio}_p${patch_size}_lp_bs_${BATCH_SIZE_PER_TPU}_${EPOCH}ep"  # a place to save checkpoints (should be under NFS)
PRETRAIN_CHKPT_DIR="/checkpoint/qianlim/mae_save/vitb_800ep_m${mask_ratio}_p${patch_size}/checkpoint-799.pth"
LOG_DIR=None # None because of a weird bug
MODEL="vit_base_patch${patch_size}"
EPOCH=90

# Default blr: 0.1, wd: 0.0, warmup: 10, see paper appendix
# Default: CLS token

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # a workaround for NFS UIDs (see "Troubleshooting")
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist \
	--tpu=${TPU_NAME} --restart-tpuvm-pod-server \
	--env XRT_MESH_CONNECT_WAIT=1200 --env PYTHONUNBUFFERED=1 -- \
python3 ${MAE_PATH}/main_linprobe.py \
        --${AGGRE_TYPE} \
	--finetune ${PRETRAIN_CHKPT_DIR} \
        --output_dir ${SAVE_DIR} \
	--log_dir ${LOG_DIR} \
	--batch_size ${BATCH_SIZE_PER_TPU} \
	--model ${MODEL} \
	--warmup_epochs 10 \
	--epochs ${EPOCH} \
	--blr 0.1 --weight_decay 0.0 \
	--data_path ${IMAGENET_DIR} \
	--num_workers 8 \
	--use_xla --resume automatic \
	2>&1 | tee $SAVE_DIR/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log

