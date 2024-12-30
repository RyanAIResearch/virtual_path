MODEL=llama2
MODEL_NAME=llama2
TASK=${TASK:-SST2}
BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
GRAD_ACC=${GRAD_ACC:-1}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-500}
OUTLIER_PERCENTAGE=${OUTLIER_PERCENTAGE:-5e-3}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=mezo-llama2-$TASK-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-TRAIN-$TRAIN-outlier-$OUTLIER_PERCENTAGE

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
    ReCoRD) 
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
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "OUTLIER_PERCENTAGE: $OUTLIER_PERCENTAGE"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "GRAD_ACC: $GRAD_ACC"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --outlier --outlier_percentage $OUTLIER_PERCENTAGE \
    --use_momentum \
    --gradient_accumulation_steps $GRAD_ACC \
    --train_as_classification \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
