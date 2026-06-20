countries=("Japan" "China" "South_Korea" "United_States" "United_Kingdom" "Germany" "Italy" "India" "Singapore" "Indonesia" "Russia" "Poland" "Romania" "Mexico" "Nigeria")
compare_countries=("China United_States Germany") # "Japan United_States Germany") #"South_Korea United_States Germany" "Japan China Germany" )

## stage=1 for question synthesis, stage=2 for rep_dist optimization

for i in "${!countries[@]}"; do
    country="${countries[i]}"
    compare_country="${compare_countries[0]}"
    echo "rep_dist question synthesis for $country with compare countries: $compare_country"
    python 1.data_synthesis.py \
        --country "$country" \
        --compare_countries $compare_country \
        --stage 1 \
        --batch_size 16 \
        --embed_model "text-embedding-ada-002_2" \
        --question_generator "trapi/gpt-5.2-reasoning" \
        --main_generator "trapi/gpt-5.3-chat" \
        --output_dir ./data # >> logs/1.data_synthesis_qwen.log
done