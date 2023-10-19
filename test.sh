python -m lm_eval \
    --model vllm \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks hellaswag \
    --limit 100 \
    --batch_size 8
