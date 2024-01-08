import pandas as pd

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
    