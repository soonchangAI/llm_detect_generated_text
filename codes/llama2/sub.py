from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import torch

MAX_LENGTH = 256

model_checkpoint = '/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect_generated_text/output/llama2-finetuned'
num_labels=2   
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,device_map='auto',torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


import pandas as pd
import numpy as np
from torch.utils.data import  DataLoader
test = pd.read_csv('/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect_generated_text/dataset/test_essays.csv')
dataset = Dataset.from_pandas(test)

def preprocess_function(examples):
    return tokenizer(examples['text'], #max_length=256,truncation=True) #$, padding=True, truncation=True
        return_tensors="pt",
        padding='longest',
        max_length=MAX_LENGTH,
        truncation=True)

test_data = dataset.map(preprocess_function, batched=True)
test_data = test_data.remove_columns(['prompt_id','text'])
test_data.set_format("torch")
test_dataloader = DataLoader(test_data, batch_size=1)


ids = []
gens = []
with torch.no_grad():
    for batch in test_dataloader:
        input_data = {}
        for k,v in batch.items():
            if k not in ['id']:
                input_data[k] = v
        
        ids.append(batch['id'][0])
        prediction = model(**input_data)
        logits = prediction.logits.numpy()
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        gens.append(probs[:,1][0])
sub = pd.DataFrame()
# print(ids)
# print(gens)
sub['id'] = ids
sub['generated'] = gens
sub.to_csv('submission.csv', index=False)
print(sub.head())