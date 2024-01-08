conda activate sgpt
export HUGGINGFACE_HUB_CACHE=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect_generated_text/checkpoints/
export TRANSFORMERS_CACHE=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect_generated_text/checkpoints/
export HF_DATASETS_CACHE=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/sparsegpt/datasets/

conda activate sgpt
export HUGGINGFACE_HUB_CACHE=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/sparsegpt/checkpoints/
export TRANSFORMERS_CACHE=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/sparsegpt/checkpoints/
export HF_DATASETS_CACHE=/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/sparsegpt/datasets/



torchrun --standalone --nproc_per_node=gpu multigpu.py 1 1

/home/cybertron/anaconda3/envs/llm/lib/python3.10/site-packages/transformers

sudo ln -s /home/cybertron/anaconda3/envs/llm/lib/python3.10/site-packages/transformers ./transformers_shortcut