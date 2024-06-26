OUTPUT=$1
OUTPUT=../../../../eval_results/new_evaluator_reward_align_model2/deberta

python eval.py \
    --split test \
    --model /home/jingyao/yifan/mycontainer/models/aligned_model2/reward/dataset_saferlhf_model_alpaca-7b-reproduced_reward_reward_model_deberta_our_data_lora_8_ppo_epochs_2_epochs_5 \
    --batch_size 4 \
    --device_id 3 \
    --output_dir $OUTPUT \
    --input_file /home/jingyao/yifan/mycontainer/eval_results/new_evaluator_reward_align_model2/llama/harmless_test_azure_dataset_saferlhf_model_alpaca-7b-reproduced_reward_reward_model_deberta_our_data_lora_8_ppo_epochs_2_epochs_5.jsonl \
    --evaluator /home/jingyao/yifan/mycontainer/models/evaluator_new_data/evaluator_deberta_tanh \
    --value_reward \
    &> $OUTPUT/new_evaluator_deberta_eval_reward_aligned_model2.log