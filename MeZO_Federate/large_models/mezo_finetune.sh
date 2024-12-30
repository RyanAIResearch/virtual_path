MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

STEPS=${STEPS:-20000}
RANK=${RANK:-16}
BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

MODE=${MODE:-ft}
EXTRA_ARGS=""

TAG=mezo-$MODEL-$TASK-$MODE-$BS-$LR-$EPS-$SEED
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    TAG=mezo-$MODEL-$TASK-$MODE-$BS-$LR-$EPS-$SEED-prefix-5
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora --lora_rank $RANK"
    TAG=mezo-$MODEL-$TASK-$MODE-$BS-$LR-$EPS-$SEED-lora-$RANK
fi

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
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "STEPS: $STEPS"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo --fp16 \
    --learning_rate $LR --per_device_train_batch_size $BS \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --train_as_classification \
    --eval_steps 500 \
    --save_steps 500 \
    --use_full_zo_update \
    --zo_eps $EPS \
    --record_time \
    --use_full_zo_update \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
