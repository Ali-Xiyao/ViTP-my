module load compiler/gcc/gcc-10.4.0-gcc-4.8.5-zept3e4 
set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=25901
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

SCRIPT_NAME=$(basename -- "$0")
# 去掉扩展名（.sh）
SCRIPT_NAME=${SCRIPT_NAME%.sh}

# 自动生成 OUTPUT_DIR
OUTPUT_DIR="work_dirs/$SCRIPT_NAME"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


TIMESTAMP=$(date +%m%d_%H%M)

NEW_SCRIPT_NAME="${SCRIPT_NAME}_${TIMESTAMP}.sh"

DATA_PATH="ViTP_configs/ft_data_medical.json"

cp "$0" "${OUTPUT_DIR}/${NEW_SCRIPT_NAME}"
cp "$DATA_PATH" "${OUTPUT_DIR}/dataset_configs_${TIMESTAMP}.json"

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
WANDB_PROJECT=ITAP \
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/home/u1120230285/lyx/InternVL/models/InternVL2_5-1B" \
  --max_steps 8000 \
  --overwrite_output_dir True \
  --TMAug_prob 0.75 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --learning_rate 2e-5 \
  --max_seq_length 4096 \
  --max_dynamic_patch 6 \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer True \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${DATA_PATH} \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 20 \
  --weight_decay 0.01 \
  --warmup_ratio 0.2 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log_${TIMESTAMP}.txt"