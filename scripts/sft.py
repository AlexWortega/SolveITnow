import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments#,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model# prepare_model_for_kbit_training
#from utils import find_all_linear_names, print_trainable_parameters

output_dir="./results-Instrucr"
model_name ="CodeLlama-7b-Instruct-hf"

dataset = load_dataset("json", data_files="leetcode-solutions.json",split="train")


base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cpu()
base_model.config.use_cache = False
#base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    #target_modules=find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
base_model.print_trainable_parameters()

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['code_with_problem'])):
        text = f"Below is an instruction that describes a task. Write a python code response that appropriately completes the request.\n\n###Instruction:{example['code_with_problem'][i].replace('python\n','\n\n### Response: python\n')}```"
        output_texts.append(text)
    return output_texts

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing =False,
    max_grad_norm= 0.3,
    num_train_epochs=3, 
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="adamw_hf",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    base_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=1024,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)