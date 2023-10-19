python3 -m lm_eval \
    --model vllm \
    --model_args pretrained=mistralai/Mistral-7B-v0.1 \
    --tasks hellaswag \
    --batch_size 1
