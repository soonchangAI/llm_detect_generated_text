## About
Fine-tuning large language models for Kaggle competition, LLM - Detect AI Generated Text


* Models
  * BERT: Single & multi-GPU data parallel
  * Llama-2-7b: multi-GPU model parallel
    
## Usage

1. BERT
* Single GPU
```
python codes/bert/single.py $num_epochs $save_every
```
* Multi-GPU
```
torchrun --standalone --nproc_per_node=gpu codes/bert/multigpu.py $num_epochs $save_every
```

2. LLAMA 2
* Train only the final layer weight because I only have access to Titan X
```
python codes/llama2/finetune_classifier.py
```
## Dependencies
* <code>torch</code> tested on v2.1.0
* <code>transformers</code> tested on v4.35.2
* <code>datasets</code>tested on v2.15.0

## To-Do

- [ ] Add Phi-2, Mistral 
## Reference:
* Code partially adapted from [PyTorch Examples](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series) and [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1)
