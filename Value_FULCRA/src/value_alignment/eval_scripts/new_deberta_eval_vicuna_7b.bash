OUTPUT=$1
OUTPUT=../../../../eval_results/new_evaluator_raw_model/deberta

python eval.py \
    --split test \
    --model lmsys/vicuna-7b-v1.5 \
    --batch_size 4 \
    --device_id 4 \
    --output_dir $OUTPUT \
    --input_file /home/jingyao/yifan/mycontainer/eval_results/new_evaluator_raw_model/llama/harmless_test_azure_vicuna-7b-v1.5.jsonl \
    --evaluator /home/jingyao/yifan/mycontainer/models/evaluator_new_data/evaluator_deberta_tanh \
    --value_reward \
    &> $OUTPUT/new_evaluator_deberta_eval_vicuna-7b-v15.log