python3 -m lm_eval \
    --model vllm \
    --model_args model=NousResearch/Nous-Hermes-llama-2-7b \
    --tasks hellaswag \
    --batch_size 1
