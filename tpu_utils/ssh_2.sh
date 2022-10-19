TPU_NAME=lpft  # !!! change to the TPU name you created
ZONE=europe-west4-a
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} #--worker 1
