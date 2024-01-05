import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import transformers
import datasets
import pandas as pd
from termcolor import colored 

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        eval_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        #print(colored('Trainer ...','green'))
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        self.model.from_pretrained(snapshot_path)
        # loc = f"cuda:{self.gpu_id}"
        # snapshot = torch.load(snapshot_path, map_location=loc)
        # self.model.load_state_dict(snapshot["MODEL_STATE"])
        # self.epochs_run = snapshot["EPOCHS_RUN"]
        # print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        output = self.model(**batch)
        loss = output.loss
        loss.backward()
        self.optimizer.step()

    def _run_eval(self, batch):

        output = self.model(**batch)
        loss = output.loss
        return loss

    def _run_epoch(self, epoch):
        #print('run_epoch')
        # print(next(iter(self.train_data)))
        b_sz = next(iter(self.train_data))['input_ids'].shape[0]
        #print(b_sz)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        count = 0
        for batch in tqdm(self.train_data):
            batch = {k:v.to(self.gpu_id) for k,v in batch.items()}
            #print(batch)
            self._run_batch(batch)
            # source = source.to(self.gpu_id)
            # targets = targets.to(self.gpu_id)
            # self._run_batch(source, targets)
            count += 1
            if count %50 == 0 and self.gpu_id == 0:
                self.eval_data.sampler.set_epoch(1)
                total_val_loss = 0
            
                with torch.no_grad():
                    for batch in self.eval_data:
                        batch = {k:v.to(self.gpu_id) for k,v in batch.items()}
                        loss = self._run_eval(batch)
                        total_val_loss += loss
            
                print('Val loss: {} for {}:'.format(count, total_val_loss))
# count = 0
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k:v.to(device) for k,v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
    def _save_snapshot(self, epoch):
        self.model.module.save_pretrained(self.snapshot_path)
        # snapshot = {
        #     "MODEL_STATE": self.model.module.state_dict(),
        #     "EPOCHS_RUN": epoch,
        # }
        # torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_snapshot(epoch)
        if self.gpu_id == 0:
            self._save_snapshot(epoch)

# def load_train_objs():
#     #train_set = MyTrainDataset(2048)  # load your dataset
#     model = torch.nn.Linear(20, 1)  # load your model
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     return train_set, model, optimizer


# def prepare_dataloader(dataset: Dataset, batch_size: int):
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         
#         sampler=DistributedSampler(dataset)
#     )


# def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
#     ddp_setup()
#     dataset, model, optimizer = load_train_objs()
#     train_data = prepare_dataloader(dataset, batch_size)
#     trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
#     trainer.train(total_epochs)
#     destroy_process_group()


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='simple distributed training job')
#     parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
#     parser.add_argument('save_every', type=int, help='How often to save a snapshot')
#     parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
#     args = parser.parse_args()
    
#     main(args.save_every, args.total_epochs, args.batch_size)
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import get_scheduler





def load_train_objs():
    num_labels = 2
    df = pd.read_csv('/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect/external_dataset/daigt-v2-train-dataset/train_v2_drcat_02.csv')

    model_checkpoint = 'bert-base-uncased'
    num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    np.random.seed(404)

    dataset = Dataset.from_pandas(df)


    def preprocess_function(examples):
        return tokenizer(examples['text'], max_length=256, padding=True, truncation=True)

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer,max_length=256)
    tokenized_datasets = dataset.map(preprocess_function,batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(['text', 'prompt_name', 'source', 'RDizzl3_seven'])
    tokenized_datasets = tokenized_datasets.rename_column("label","labels")
    tokenized_datasets.set_format("torch")
    split_datasets = tokenized_datasets.train_test_split(test_size=0.01)

    train_dataloader = DataLoader(split_datasets['train'], 
                    batch_size=16,
                    pin_memory=True,
                    shuffle=False,
                    sampler=DistributedSampler(split_datasets['train'])
                    )
    eval_dataloader = DataLoader(split_datasets['test'], batch_size=16,pin_memory=True,shuffle=False,
                        sampler=DistributedSampler(split_datasets['test']))
    
    
 #, collate_fn=data_collator)

    num_labels=2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    #optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=5e-5)
    # train_set, model, optimizer
    return train_dataloader, eval_dataloader, model, tokenizer, optimizer

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect/codes/bert-finetuned"):
    ddp_setup()
    train_dataloader, eval_dataloader,model, tokenizer, optimizer = load_train_objs()
    # print('Train_dataloader')
    # print(train_dataloader)
    # train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_dataloader,eval_dataloader, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()
    #train_set = MyTrainDataset(2048)  # load your dataset
    # model = torch.nn.Linear(20, 1)  # load your model
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # return train_set, model, optimizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size)
# 

# for k,v in model.named_parameters():
#     # print(k,v.shape)
#     if 'pooler' in k or 'classifier' in k:
#         v.requires_grad = True 
#     else:
#         v.requires_grad = False

# for k,v in model.named_parameters():
#     print(k,v.requires_grad)

# num_epochs = 1
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,)
# print(num_training_steps)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)


# progress_bar = tqdm(range(num_training_steps))

# model.train()

# count = 0
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k:v.to(device) for k,v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         count += 1
#         if count %50 == 0:
#             total_val_loss = 0
#             with torch.no_grad():
#                 for batch in eval_dataloader:
#                     batch = {k:v.to(device) for k,v in batch.items()}
#                     outputs = model(**batch)
#                     loss = outputs.loss
#                     total_val_loss += loss.item()
#             print('Val loss :',total_val_loss)

#         progress_bar.update(1)

# model.save_pretrained('/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect/codes/bert-base-uncased-finetuned')
# tokenizer.save_pretrained('/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/llm_detect/codes/bert-base-uncased-finetuned')
