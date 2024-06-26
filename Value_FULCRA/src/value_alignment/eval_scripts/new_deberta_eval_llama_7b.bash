OUTPUT=$1
OUTPUT=../../../../eval_results/new_evaluator_raw_model/deberta

python eval.py \
    --split test \
    --model meta-llama/Llama-2-7b-chat-hf \
    --batch_size 4 \
    --device_id 3 \
    --output_dir $OUTPUT \
    --input_file /home/jingyao/yifan/mycontainer/eval_results/new_evaluator_raw_model/llama/harmless_test_azure_Llama-2-7b-chat-hf.jsonl \
    --evaluator /home/jingyao/yifan/mycontainer/models/evaluator_new_data/evaluator_deberta_tanh \
    --value_reward \
    &> $OUTPUT/new_evaluator_deberta_eval_llama2-7b-chat.log