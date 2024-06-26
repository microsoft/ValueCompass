OUTPUT=$1
OUTPUT=.

python eval.py \
    --split test \
    --api mistral \
    --model mistral-large-latest \
    --batch_size 4 \
    --device_id 0 \
    --output_dir $OUTPUT \
    --evaluator /home/jingyao/projects/Alignment/Value_Benchmark/output/evaluator/deberta-v3-large_tanh \
    --reward_model OpenAssistant/reward-model-deberta-v3-large-v2 \
    --value_reward \
    --reward

# --input_file ./harmless_test_mistral_mistral-large-latest.jsonl \