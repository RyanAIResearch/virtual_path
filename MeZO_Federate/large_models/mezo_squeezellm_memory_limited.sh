MODEL=${MODEL:-NousResearch/Llama-2-7b-hf}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-32}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-500}
SQUEEZELLM_CKPT=${SQUEEZELLM_CKPT:-/share/desa/nfs02/wg247/SqueezeLLM/nuq/llama2-7b/3-bit-0-1e-3.pt}
SQUEEZELLM_BITS=${SQUEEZELLM_BITS:-3}

TAG=mezo-squeezellm-$MODEL_NAME-$SQUEEZELLM_BITS-$STEPS-$BS-$LR-$EPS-$SEED-memory-limited

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
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "squeezellm ckpt: $SQUEEZELLM_CKPT"
echo "squeezellm wbits: $SQUEEZELLM_BITS"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --use_squeezellm --squeezellm_ckpt $SQUEEZELLM_CKPT --squeezellm_wbits $SQUEEZELLM_BITS \
    --memory_limit_scenario \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
