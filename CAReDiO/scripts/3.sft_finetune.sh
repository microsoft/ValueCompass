countries=("Japan")
sample_nums=(2000 1000 500 200)
rounds=("0" "1" "2")
optimize_methods=("general" "rep_dist" "rep_dist")
select_methods=("random" "rep_dist" "rep_dist")
backbone="qwen/qwen3-8b"
model_dir="qwen3-8b"
data_source="qwen3-8b"
align_methods=("None" "Role-Play")
train_data_type="completion"

for sample_num in "${sample_nums[@]}"; do
    for i in "${!rounds[@]}"; do
        round="${rounds[i]}"
        optimize_method="${optimize_methods[i]}"
        select_method="${select_methods[i]}"
        for country in "${countries[@]}"; do
            for align_method in "${align_methods[@]}"; do
                echo "Finetuning cultural model with $backbone for $country with $sample_num samples optimized by $optimize_method & selected by $select_method, using align method $align_method, train data type $train_data_type, $data_source, optimization round $round"
                python 3.sft_finetune.py \
                    --align_method="${align_method}" \
                    --train_data_type="${train_data_type}" \
                    --model_name="${backbone}" \
                    --dataset_name="./data/$country/${optimize_method}_${data_source}_cluster_threshold_0.15_round_${round}/${select_method}_${sample_num}.jsonl" \
                    --output_dir="./outputs/${model_dir}/$country/sft/align_${align_method}/${optimize_method}_${data_source}_round_${round}_data_${select_method}_data_type_${train_data_type}_sample_num_${sample_num}" \
                    --run_name="sft_${model_dir}_our_${country}_${sample_num}" \
                    --use_bnb=False \
                    --seq_length=800 \
                    --num_train_epochs=10 \
                    --max_steps=-1 \
                    --logging_steps=10 \
                    --save_steps=50 \
                    --per_device_train_batch_size=4 \
                    --per_device_eval_batch_size=1 \
                    --gradient_accumulation_steps=8 \
                    --gradient_checkpointing=False \
                    --group_by_length=False \
                    --learning_rate=1e-4 \
                    --lr_scheduler_type="cosine" \
                    --warmup_steps=100 \
                    --weight_decay=0.05 \
                    --optim="paged_adamw_32bit" \
                    --bf16=True \
                    --remove_unused_columns=False \
                    --report_to="tensorboard"
            done
        done
    done
done

# --dataset_name="../data/$country/${optimize_method}_${model_dir}/${select_method}_${sample_num}.jsonl"
# backbone="meta-llama/Llama-3.1-8B-Instruct"
