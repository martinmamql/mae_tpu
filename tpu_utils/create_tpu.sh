TPU_NAME=mae #lpft
ZONE=europe-west4-a  # a location where we have available TPU quota
ACCELERATOR_TYPE=v3-128
STARTUP_SCRIPT=tpu_start_mae.sh  # !!! change to your startup script path

RUNTIME_VERSION=tpu-vm-pt-1.10  # this is the runtime we use for PyTorch XLA (it contains PyTorch 1.10)

# create a TPU VM (adding `--reserved` to create reserved TPUs)
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} \
  --version ${RUNTIME_VERSION} \
  --metadata-from-file=startup-script=${STARTUP_SCRIPT}

