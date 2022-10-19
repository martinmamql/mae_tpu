TPU_NAME_LIST="mae"  # !!! change to the list of TPU VMs you're using
ZONE=europe-west4-a

PYTHON_PROCESS_NUM=8  # a busy node should have at least 8 python processes (for 8 TPU cores)

for TPU_NAME in $TPU_NAME_LIST; do
	out=$(gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker 0 --command "pgrep python | wc -l" 2>/dev/null)
	if [ -z $out ]; then
		STATUS="DOWN"
	elif [[ $out -ge $PYTHON_PROCESS_NUM ]]; then
		STATUS="BUSY"
	elif [[ $out -lt $PYTHON_PROCESS_NUM ]]; then
		STATUS="IDLE"
	else
		STATUS="UNKN"
	fi
	echo $STATUS $TPU_NAME
done

