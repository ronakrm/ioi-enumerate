poetry run python get_logits.py \
    --model "gpt2-small" \
    --res_file "small_logits.csv" \
    --max_num_samples 10000000 \
    --batch_size 1024 \
    --device "cuda"

poetry run python get_logits.py \
    --model "gpt2-medium" \
    --res_file "medim_logits.csv" \
    --max_num_samples 10000000 \
    --batch_size 512 \
    --device "cuda"

poetry run python get_logits.py \
    --model "gpt2-large" \
    --res_file "large_logits.csv" \
    --max_num_samples 10000000 \
    --batch_size 128 \
    --device "cuda"
