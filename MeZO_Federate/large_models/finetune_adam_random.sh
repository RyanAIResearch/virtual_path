MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

EPOCH=${EPOCH:-10}
BS=${BS:-8}
LR=${LR:-1e-5}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
PERCENTAGE=${PERCENTAGE:-1e-3}
GA=${GA:-8}

MODE=${MODE:-ft}
TAG=$MODE-$EPOCH-$BS-$GA-$LR-$SEED-random-$PERCENTAGE

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "EPOCH: $EPOCH"
echo "BS: $BS"
echo "LR: $LR"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Gradient accumulation: $GA"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --trainer regular --fp16 \
    --max_steps 1000 \
    --learning_rate $LR --per_device_train_batch_size $BS \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps 200 --save_steps 200 \
    --train_as_classification \
    --use_adam \
    --random_subset_weights --outlier_percentage $PERCENTAGE \
    --gradient_accumulation_steps $GA \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"