MODEL=${MODEL:-facebook/opt-1.3b}
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
SQUEEZELLM_CKPT=${SQUEEZELLM_CKPT:-/share/desa/nfs02/wg247/SqueezeLLM/model/opt-1.3b/3bit.pt}
SQUEEZELLM_BITS=${SQUEEZELLM_BITS:-3}

LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
PREFIX_LENGTH=${PREFIX_LENGTH:-5}

MODE=${MODE:-lora}


TAG=mezo-squeezellm-peft-$MODEL_NAME-$SQUEEZELLM_BITS-$STEPS-$BS-$LR-$EPS-$SEED
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix $PREFIX_LENGTH --no_reparam --prefix_init_by_real_act"
    TAG=mezo-peft-$MODEL_NAME-$SQUEEZELLM_BITS-$STEPS-$BS-$LR-$EPS-$SEED-prefix-$PREFIX_LENGTH
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA"
    TAG=mezo-peft-$MODEL_NAME-$SQUEEZELLM_BITS-$STEPS-$BS-$LR-$EPS-$SEED-lora-$LORA_RANK
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
    --use_squeezellm_peft --squeezellm_ckpt $SQUEEZELLM_CKPT --squeezellm_wbits $SQUEEZELLM_BITS \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
