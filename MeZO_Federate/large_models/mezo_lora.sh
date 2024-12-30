MODEL=${MODEL:-llama2}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

STEPS=${STEPS:-3000}
RANK=${RANK:-16}
BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}

MODE=${MODE:-lora}
EXTRA_ARGS=""

EXTRA_ARGS="--lora --lora_rank $RANK"
TAG=mezo-$MODEL-$TASK-$MODE-$BS-$LR-$EPS-$SEED-lora-$RANK
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
    MultiRC) # Can only fit real bsz = 2 on 80G A100
        # GA=$(expr $BS / 2)
        # BS=2
        # echo "Gradient accumulation: $GA"
        # TASK_ARGS="--gradient_accumulation_steps $GA"
        ;;
    ReCoRD) # Can only fit real bsz = 2 on 80G A100
        GA=$(expr $BS / 2)
        BS=2
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
        ;;
    DROP) # Can only fit real bsz = 1 on 80G A100
        GA=$(expr $BS / 1)
        BS=1
        echo "Gradient accumulation: $GA"
        TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
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
    --trainer zo \
    --learning_rate $LR --per_device_train_batch_size $BS \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --train_as_classification \
    --load_bfloat16 \
    --save_steps 200 \
    --eval_steps 200 \
    --use_momentum \
    --zo_eps $EPS \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
