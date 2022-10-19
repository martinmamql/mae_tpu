#!/bin/bash
TPU_NAME=mae # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm scp --zone ${ZONE} --worker 0 ${TPU_NAME}:/checkpoint/qianlim/mae_save/vitb_800ep_m0.25_p16/checkpoint_799.pth .

