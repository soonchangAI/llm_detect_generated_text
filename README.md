## About
Fine-tuning large language models for Kaggle competition, LLM - Detect AI Generated Text

## Features
* Distributed data parallel training on single node, multi-GPU
## Models
* BERT
## Usage

```
torchrun --standalone --nproc_per_node=gpu multigpu.py $num_epochs $save_every
```
## Dependencies
* <code>torch</code> tested on v2.1.0
* <code>transformers</code> tested on v4.35.2
* <code>datasets</code>tested on v2.15.0
## Reference:
* Code partially adapted from [PyTorch Examples](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series) and [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1)
