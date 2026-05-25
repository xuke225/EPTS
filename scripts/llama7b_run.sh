CUDA_VISIBLE_DEVICES=0,1 python precomputer_wanda_importance.py \
    --model decapoda-research/llama-7b-hf \
    --model_type llama \
    --prune_method wanda \
    --importance_dir "path to save wanda importance"

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model decapoda-research/llama-7b-hf \
    --model_type llama \
    --prune_method wanda \
    --fusion_level datasets \
    --nsamples 128 \
    --importance_dir "path to save wanda importance" \
    --reconstructed_model_path "Path to save the reconstructed model" 2>&1 | tee output.log

CUDA_VISIBLE_DEVICES=0,1 python inference.py \
   --model "decapoda-research/llama-7b-hf" \
   --prune_rate 0.7 \
   --importance_dir "path to save wanda importance" \
   --reconstructed_model_path "Path to save the reconstructed model"