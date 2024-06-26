OUTPUT=$1
OUTPUT=.
# export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval.py \
    --data harmless \
    --split test \
    --api mistral \
    --model mistral-large-latest \
    --batch_size 1 \
    --device_id 0 \
    --output_dir $OUTPUT \
    --inference # \
    # &> $OUTPUT/new_evaluator_llama_eval_mistral_large_latest.log

    # --reward \
  