poetry run python get_logits.py \
    --model_name "gpt2-small" \
    --res_file "results/small_logits.csv" \
    --max_num_samples 10000000 \
    --batch_size 1024 \
    --device "cuda"

poetry run python get_logits.py \
    --model_name "gpt2-medium" \
    --res_file "results/medium_logits.csv" \
    --max_num_samples 10000000 \
    --batch_size 512 \
    --device "cuda"

poetry run python get_logits.py \
    --model_name "gpt2-large" \
    --res_file "results/large_logits.csv" \
    --max_num_samples 10000000 \
    --batch_size 128 \
    --device "cuda"
