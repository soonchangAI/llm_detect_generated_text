from transformers import AutoModelForCausalLM, AutoTokenizer
model_checkpoint = '/media/cybertron/fa54fcb6-b5e1-492e-978a-6389519c168a/sparsegpt/output/phi-1.5'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint) #, device_map='auto') #,trust_remote_code=True)

for k,v in model.named_parameters():
    print(k,v.shape)