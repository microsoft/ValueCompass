OUTPUT=$1
OUTPUT=../../../../eval_results/new_evaluator_large_reward_align_model/deberta

python eval.py \
    --split test \
    --model /home/jingyao/yifan/mycontainer/models/aligned_model/ppo/dataset_saferlhf_model_alpaca-7b-reproduced_reward_reward-model-deberta-v3-large-v2lora_8/final_model \
    --batch_size 1 \
    --device_id 1 \
    --output_dir $OUTPUT \
    --input_file /home/jingyao/yifan/mycontainer/eval_results/old_evaluator_reward_align_model/harmless_test_azure_final_model.jsonl \
    --evaluator /home/jingyao/yifan/mycontainer/models/evaluator_new_data/evaluator_deberta_tanh \
    --value_reward \
    &> $OUTPUT/new_evaluator_deberta_eval_large_reward_aligned_model.log