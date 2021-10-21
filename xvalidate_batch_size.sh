#!/bin/bash
for BATCH_SIZE in 16 32 64 128
do
	for MODEL in "slabs_weighted_bal" "slabs_unweighted_two_way"
	do
		echo "# ----- $MODEL $BATCH_SIZE ----#"
		taskset --cpu-list 0-30 python -m experiments.waterbirds  \
			--model_to_tune $MODEL --pick_best  \
			--experiment_name "8090" --clean_back 'True' --batch_size $BATCH_SIZE
	done
done
