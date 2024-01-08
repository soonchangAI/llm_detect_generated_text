from datasets import Dataset
import transformers 
from typing import Dict, Sequence
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import LlamaModel
from torch.utils.data import DataLoader
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from termcolor import colored 
DEBUG = False
MAX_LENGTH = 256

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :MAX_LENGTH] #self.tokenizer.model_max_length] 
        labels = torch.tensor(labels)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch
        
def load_train_objs():
    num_labels = 2
    df = pd.read_csv('/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect_generated_text/external_dataset/daigt-v2-train-dataset/train_v2_drcat_02.csv')
    model_checkpoint = 'meta-llama/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    print("Padding token id :", tokenizer.pad_token_id)
    if DEBUG:
        model = 0 
        
    else:
        config = AutoConfig.from_pretrained(model_checkpoint)
        model = LlamaModel(config) #,torch_dtype=torch.float16)
        print(model)
        model.config.pad_token_id = tokenizer.unk_token_id                                                    
    for k,v in model.named_parameters():
        print(k,v.shape)

    np.random.seed(404)

    dataset = Dataset.from_pandas(df)


    def preprocess_function(examples):
        return tokenizer(examples['text'], #max_length=256,truncation=True) #$, padding=True, truncation=True
            return_tensors="pt",
            padding='longest',
            max_length=MAX_LENGTH,
            truncation=True)
        #return tokenizer(examples['text'], max_length=256, padding=True, truncation=True)

    tokenized_datasets = dataset.map(preprocess_function,batched=True)
    print(tokenized_datasets)
    print(tokenized_datasets[0])
    tokenized_datasets = tokenized_datasets.remove_columns(['text', 'prompt_name', 'source', 'RDizzl3_seven'])
    tokenized_datasets = tokenized_datasets.rename_column("label","labels")
    tokenized_datasets.set_format("torch")
    split_datasets = tokenized_datasets.train_test_split(test_size=0.01)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(split_datasets['train'], 
                    batch_size=4,
                    pin_memory=True,
                    collate_fn= data_collator,
                    shuffle=False)
    eval_dataloader = DataLoader(split_datasets['test'], 
                                    batch_size=4,
                                    pin_memory=True,
                                    collate_fn=data_collator,
                                    shuffle=False)
    
    if DEBUG:
        optimizer = 0 # 
    else: 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=5e-5)

    return train_dataloader, eval_dataloader, model, tokenizer, optimizer



train_dataloader, eval_dataloader, model, tokenizer, optimizer = load_train_objs()


del eval_dataloader



