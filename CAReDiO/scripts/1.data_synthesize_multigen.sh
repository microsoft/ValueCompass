countries=("Japan" "China" "South_Korea" "United_States" "United_Kingdom" "Germany" "Italy" "India" "Singapore" "Indonesia" "Russia" "Poland" "Romania" "Mexico" "Nigeria")
compare_countries=("China United_States Germany") # "Japan United_States Germany") #"South_Korea United_States Germany" "Japan China Germany" )

## stage=1 for question synthesis, stage=2 for rep_dist optimization

for i in "${!countries[@]}"; do
    country="${countries[i]}"
    compare_country="${compare_countries[i]}"
    echo "rep_dist question synthesis for $country with compare countries: $compare_country"
    python 1.data_synthesis.py \
        --country "$country" \
        --compare_countries $compare_country \
        --stage 2 \
        --batch_size 2 \
        --embed_model "text-embedding-ada-002_2" \
        --question_generator "trapi/gpt-5.2-reasoning" \
        --main_generator "trapi/deepseek-v4-pro" \
        --multi_generators "trapi/deepseek-v4-pro" "trapi/gpt-5.2-reasoning" "trapi/grok-4.20"  "trapi/kimi-k2.6" \
        --output_dir ./data # >> logs/1.data_synthesis_qwen.log
done

# "trapi/gpt-5.3-chat" 