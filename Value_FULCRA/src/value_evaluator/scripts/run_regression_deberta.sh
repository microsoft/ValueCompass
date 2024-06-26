OUTPUT=../../output/evaluator/regression
mkdir -p $OUTPUT

deepspeed --num_gpus=1 ./main.py \
    --train \
    --model_type regression \
    --class_num 3 \
    --cache_dir ~/.cache/huggingface/ \
    --dataset saferlhf \
    --data_path  ../../data/evaluator_data.jsonl \
    --value_item_or_type value_type \
    --value_type_dimension 10 \
    --label_num 100 \
    --max_new_tokens 256 \
    --model_name_or_path microsoft/deberta-v3-large \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_seq_len 2048 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 50 \
    --output_dir $OUTPUT \
    --seed 2023 \
    --gradient_checkpointing \
    --zero_stage 0 \
    --lora_module_name "layers."\
    --lora_dim 8 \
    --print_loss \
    --print_every_n_step 50 \
    --deepspeed > $OUTPUT/evaluator_regression.log

# --train \
# meta-llama/Llama-2-7b-chat-hf
# facebook/opt-1.3b
# bert-large-uncased