MAE_PATH=/checkpoint/qianlim/workspace/mae_tpu  # where the repo is cloned above
IMAGENET_DIR=/checkpoint/imagenet-1k/  # where ImageNet-1k is downloaded above
mask_ratio=$1
patch_size=$2
SAVE_DIR="/checkpoint/qianlim/mae_save/vitb_800ep_m${mask_ratio}_p${patch_size}"  # a place to save checkpoints (should be under NFS)
MODEL="mae_vit_base_patch${patch_size}"
EPOCH=800
TPU_NAME=mae  # !!! change to the TPU name you created
BATCH_SIZE_PER_TPU=32  # 4096 (total batch size) // 128 (tpu cores)

sudo mkdir -p $SAVE_DIR && sudo chmod -R 777 $SAVE_DIR  # a workaround for NFS UIDs (see "Troubleshooting")
cd ${HOME} && python3 -m torch_xla.distributed.xla_dist \
	--tpu=${TPU_NAME} --restart-tpuvm-pod-server \
	--env XRT_MESH_CONNECT_WAIT=1200 --env PYTHONUNBUFFERED=1 -- \
python3 ${MAE_PATH}/main_pretrain.py \
        --output_dir ${SAVE_DIR} \
	--log_dir ${SAVE_DIR} \
	--batch_size ${BATCH_SIZE_PER_TPU} \
	--model ${MODEL} \
	--norm_pix_loss \
	--mask_ratio $mask_ratio \
	--epochs ${EPOCH} \
	--warmup_epochs 40 \
	--blr 1.5e-4 --weight_decay 0.05 \
	--data_path ${IMAGENET_DIR} \
	--num_workers 8 \
	--use_xla --resume automatic \
	2>&1 | tee $SAVE_DIR/stdout_stderr_$(date +%Y-%m-%d_%H-%M-%S).log

