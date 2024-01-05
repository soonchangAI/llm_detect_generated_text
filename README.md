## About
Fine-tuning large language models for Kaggle competition, LLM - Detect AI Generated Text

## Features
* Distributed data parallel training on single node, multi-GPU
## Models
* BERT
## Run

```
torchrun --standalone --nproc_per_node=gpu multigpu.py $num_epochs $save_every
```
## Reference:
* Code partially adapted from [PyTorch Examples](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series) and [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1)
