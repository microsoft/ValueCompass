OUTPUT=$1
OUTPUT=../../eval_results/base_alpaca_evaluator_deberta_tanh_target_ideal

python eval.py \
    --split test \
    --data harmless \
    --model ../../output/value_alignment/dataset_saferlhf_model_alpaca-7b-reproduced_reward_deberta_tanh_ensemble_bs_32_kl_0.1_ppo_epochs_2_epochs_20_ideal/final_model/ \
    --batch_size 4 \
    --device_id 0 \
    --output_dir $OUTPUT \
    --inference # &> $OUTPUT/inference.log

# --model ../../output/value_alignment/dataset_saferlhf_model_alpaca-7b-reproduced_reward_deberta_tanh_ensemble_bs_32_kl_0.1_ppo_epochs_2_epochs_20_ideal/final_model/ \
# PKU-Alignment/alpaca-7b-reproduced \