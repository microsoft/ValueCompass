OUTPUT=$1
OUTPUT=.
# export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval.py \
    --split test \
    --model baichuan-inc/Baichuan2-7B-Chat \
    --batch_size 1 \
    --device_id 0 \
    --output_dir $OUTPUT \
    --inference \
    &> $OUTPUT/new_evaluator_llama_eval_baichuan2-7b-chat.log

    # --reward \
  