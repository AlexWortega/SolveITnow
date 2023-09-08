import torch
import re
from multiprocessing import Pool
import wandb
from transformers import GPT2TokenizerFast, GPT2LMHeadModel#, CodeLlamaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import  AdamW, get_linear_schedule_with_warmup
from torch import optim
SEED = 27

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
MODEL_NAME =  "CodeLlama-7b-hf" # берем именно ее, у нее правильно выставлены BOS, EOS токены

import transformers
import accelerate
#from transformers.utils.bitsandbytes import replace_with_bnb_linear
import tensor_parallel as tp

tokenizer = transformers.LlamaTokenizer.from_pretrained(MODEL_NAME)
#special_tokens_dict = {'additional_special_tokens': ['user:', 'bot:']}
#tokenizer.add_special_tokens({'eos_token': '<instructionE>'})
#tokenizer.add_special_tokens({'bos_token': '<instructionE>'})
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#tokenizer.add_special_tokens(special_tokens_dict)


model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME,  torch_dtype=torch.float16)

model.resize_token_embeddings(len(tokenizer))

#model = tp.TensorParallelPreTrainedModel( model, device_ids=["cuda:2", "cuda:3"],)
    
#model = replace_with_bnb_linear(model, quantization_config=transformers.utils.quantization_config.BitsAndBytesConfig(load_in_8bit=True))
#model.is_loaded_in_8bit = True

from accelerate import Accelerator

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

import wandb
api_key = 'e461a6a3bca9f7cec3390a40dc10cdf576ce3252'
wandb.login(key=api_key)

wandb.init(project='LLama',name='kaggle_lora')


from datasets import load_dataset
 
dataset = load_dataset("json",data_files= "leetcode-solutions.json")

import torch
from multiprocessing import Process, Queue, Manager


import torch
from torch.utils.data import Dataset,DataLoader
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor

class rulm_Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        
        
        i = 1000000000
        self.tokenized = []
        
        for q in tqdm(dataset):
                    try:
                        if i>0:
                            #print(q)
                            pr = q["code_with_problem"]
                            encoded_pr = self._encode(text=pr, tokenizer=tokenizer)

                            i=i-1
                            self.tokenized.append(encoded_pr)
                        else:
                            break
                    except Exception as e:
                        print(e)
                        pass
        #c = torch.cat(self.tokenized,1)
        self.samples = self.tokenized
       

        
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return (self.samples[item])

    def _encode(self, text, tokenizer):
        encoded_sample = tokenizer.encode(text, max_length=1024, truncation=True, add_special_tokens=False,  return_tensors='pt')
        return encoded_sample





concatenated_dataset = rulm_Dataset(dataset['train'], tokenizer)



from torch.utils.data import DataLoader
loader = DataLoader(concatenated_dataset, batch_size=1, num_workers=0, shuffle=True)


from transformers import  AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
lr = 2e-5

optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))

max_steps = 10000



scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, div_factor=25, pct_start=0.2)

gradient_accumulation_steps = 128
from accelerate.utils import GradientAccumulationPlugin

gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps)

accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_plugin=gradient_accumulation_plugin)

device = accelerator.device

model = model.to(device)

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, loader, scheduler
)

import time
i = 0
num_tokens = 0
for k in range(3):
    
   
    
    max_norm = 0.1
    for input_ids in tqdm(training_dataloader):
        i+=1
        
        with accelerator.autocast():


            start_time = time.time()
            loss = model(input_ids=input_ids.squeeze(1), labels=input_ids.squeeze(1)).loss
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            #if accelerator.sync_gradients:
                   # accelerator.clip_grad_value_(model.parameters(), max_norm)

            loss = loss / gradient_accumulation_steps


            end_time = time.time()

            fw_time = end_time-start_time




            start_time = time.time()
            accelerator.backward(loss)
            end_time = time.time()

            backward_time = end_time-start_time
                # wandb.log({'backward_time':end_time-start_time})



            num_tokens += 1024*gradient_accumulation_steps


            wandb.log({'loss': loss.item()*gradient_accumulation_steps,'learning_rate': optimizer.param_groups[0]['lr'], 'fw_time':fw_time, "backward_time":backward_time, "num_toks":num_tokens})

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


            if i % 1000 == 0:
                try:
                    state_dict = model.state_dict()

                    accelerator.save(state_dict, f"solve_now/sft_ll{i}.pt")


                except:
                     pass
