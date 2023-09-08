import sys 
import argparse
import logging
import os
import gc
import string
import io 
import re
import pickle
import random
import torch
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from model import Model
from datetime import datetime
from datasets import Dataset
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)
import sys

# создаем логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

def convert_examples_to_features(nl, code, tokenizer, code_length=256, nl_length = 128):
    code_tokens = tokenizer.tokenize(code)[:code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    
    nl_tokens = tokenizer.tokenize(nl)[:nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length
    return code_ids, nl_ids

def generate_and_tokenize_prompt(data_point, tokenizer):
    code = data_point["code"]
    nl = data_point["request"]
    code_ids, nl_ids = convert_examples_to_features(nl, code, tokenizer)
    return {"code_ids":code_ids, "nl_ids":nl_ids}

# Функция для обработки текста
def process_text(text):
    # Разделение слов по заглавной букве в середине слова
    text = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', text)
    # Разделение слов по предпоследней заглавной букве при наличии нескольких заглавных букв подряд
    text = re.sub(r'([А-ЯЁA-Z]+)([А-ЯЁA-Z][а-яёa-z]+)', r'\1 \2', text)
    # Преобразование текста в нижний регистр
    text = text.lower()
    return text

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()       

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_eval_mrr(train_data, args, model):
    code_vecs = [] 
    nl_vecs = []
    for i in tqdm(range(len(train_data))):
        code_inputs = torch.tensor([train_data[i]['code_ids']]).to(args.device)
        nl_inputs = torch.tensor([train_data[i]['nl_ids']]).to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy())
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    scores = np.matmul(nl_vecs,code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    code_urls = [j for j in range(sort_ids.shape[0])]
    ranks = []
    for i in tqdm(range(sort_ids.shape[0])):
        sort_id = sort_ids[i]
        rank = 0
        find = False
        for idx in sort_id:
            if find is False:
                rank += 1
            if code_urls[idx] == i:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
            print(ranks)
    
    result = {
            "eval_mrr":float(np.mean(ranks))
        }
    return result

def check_p_dog_k(train_data, args, model, k=5):
    code_vecs = [] 
    nl_vecs = []
    for i in tqdm(range(len(train_data))):
        code_inputs = torch.tensor([train_data[i]['code_ids']]).to(args.device)
        nl_inputs = torch.tensor([train_data[i]['nl_ids']]).to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy())
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    scores = np.matmul(nl_vecs,code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    code_urls = [j for j in range(sort_ids.shape[0])]
    ranks = []
    for i in tqdm(range(sort_ids.shape[0])):
        sort_id = sort_ids[i]
        
        find = False
        for idx in sort_id[:k]:
            if find is False:
                pass
            if code_urls[idx] == i:
                find = True
        if find:
            ranks.append(1)
        else:
            ranks.append(0)
    
    result = {
            f"p@{k}":float(np.sum(ranks)/len(train_data))
        }
    return result

def load_data(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == '.pkl':
        return pd.read_pickle(file_path)
    elif extension == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f'Unsupported file extension: {extension}. Only .pkl and .csv are supported.')

def date():
    # Получаем текущую дату
    current_date = datetime.now().date()
    
    # Преобразуем дату в строку
    date_string = current_date.strftime('%Y-%m-%d')
    return date_string


def remove_words_above_threshold(sentence, threshold, keep_first=False):
    # Преобразовываем предложение в список слов
    words = sentence.split()

    # Инициализируем словарь для подсчета частотности слов
    frequency = {}

    # Итерируем по списку слов
    for word in words:
        if word not in frequency:
            # Если слово встречается впервые, инициализируем его счетчик единицей
            frequency[word] = 1
        else:
            # Если слово уже встречалось, увеличиваем его счетчик на единицу
            frequency[word] += 1

    # Удаляем слова, частотность которых превышает заданный порог
    # Если keep_first = True, мы оставляем первое вхождение слова
    if keep_first:
        word_set = set()
        words = [word for word in words if (frequency[word] <= threshold) or (word not in word_set and not word_set.add(word))]
    else:
        words = [word for word in words if frequency[word] <= threshold]
    
    # Возвращаем обновленное предложение
    return ' '.join(words)

def remove_specific_words(text):
    words_to_remove = ["возвращает", "функция", "процедура", "получает", "формирует", "заполняет", "выполняет", "определяет"]
    # Преобразуем текст в нижний регистр и удаляем знаки пунктуации
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered_words = [word for word in words if word not in words_to_remove]
    cleaned_text = " ".join(filtered_words)
    return cleaned_text

def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset

    df = load_data(args.train_dataset)
    df_val = load_data(args.val_dataset)

    if args.space_aug:
        df['request'] = df['request'].apply(aug_comment_short)
        df['code'] = df['code'].apply(aug_code)
        df_val['request'] = df_val['request'].apply(aug_comment_short)
        df_val['code'] = df_val['code'].apply(aug_code)

    if args.frequency_aug:
        df['request'] = df['request'].apply(lambda x: remove_words_above_threshold(x, args.frequency_threshold, args.frequency_keep_first))
        df_val['request'] = df_val['request'].apply(lambda x: remove_words_above_threshold(x, args.frequency_threshold, args.frequency_keep_first))
    # вот это пробный вариант
    if args.frequency_all_text_aug:
        df_copy = df.copy()
        df_val_copy = df_val.copy()
        df_copy['request'] = df['request'].apply(remove_specific_words)
        df_val_copy['request'] = df_val['request'].apply(remove_specific_words)
        df = pd.concat([df, df_copy], axis=0)
        df_val = pd.concat([df_val, df_val_copy], axis=0)
    #до этого 
    dataset = Dataset.from_pandas(df)
    train_data = (
        dataset.map(generate_and_tokenize_prompt, fn_kwargs={'tokenizer': tokenizer})
    )
    
    df, train_subsample = train_test_split(df, test_size=0.2, random_state=args.seed)
    dataset_train_subsample = Dataset.from_pandas(train_subsample)
    train_subsample_data = (
        dataset_train_subsample.map(generate_and_tokenize_prompt, fn_kwargs={'tokenizer': tokenizer})
    )

    dataset_val = Dataset.from_pandas(df_val)
    val_data = dataset_val.map(generate_and_tokenize_prompt, fn_kwargs={'tokenizer': tokenizer})
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    model.zero_grad()

    cleanup()

    metrics_path = './metrics/'

    if not os.path.exists(metrics_path):
        os.makedirs('metrics')

    # Создаем файл
    metrics_file = os.path.join(metrics_path, f'{date()}.txt')
    with open(metrics_file, 'w') as file:
        file.write('')  # Пишем пустую строку в файл
            
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in tqdm(range(args.num_train_epochs)): 
        model.train()
        for step, batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = torch.stack(batch["code_ids"], dim=0).to(args.device)    
            nl_inputs = torch.stack(batch["nl_ids"], dim=0).to(args.device)
            #get code and nl vectors
            code_inputs = code_inputs.t()
            nl_inputs = nl_inputs.t()
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
        
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            scores = scores*20
            loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            logger.debug(f"Epoch {idx} Step {step} - Batch loss: {loss.item():.4f}")
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                print("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
        
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        model.eval()
        val_mrr = check_eval_mrr(train_subsample_data, args, model)
        val_p_to_1 = check_p_dog_k(train_subsample_data, args, model, k=1)
        val_p_to_3 = check_p_dog_k(train_subsample_data, args, model, k=3)
        val_p_to_5 = check_p_dog_k(train_subsample_data, args, model, k=5)
        val_p_to_10 = check_p_dog_k(train_subsample_data, args, model, k=10)
        val_p_to_15 = check_p_dog_k(train_subsample_data, args, model, k=15)
        eval_mrr = check_eval_mrr(val_data, args, model)
        eval_p_to_1 = check_p_dog_k(val_data, args, model, k=1)
        eval_p_to_3 = check_p_dog_k(val_data, args, model, k=3)
        eval_p_to_5 = check_p_dog_k(val_data, args, model, k=5)
        eval_p_to_10 = check_p_dog_k(val_data, args, model, k=10)
        eval_p_to_15 = check_p_dog_k(val_data, args, model, k=15)
        # Открываем текстовый файл для записи 
        with io.open(metrics_file, "a", encoding="utf-8") as f: 
            f.write(f"Train Epoch {idx} MRR {val_mrr}")
            f.write('\n')
            f.write(f"Train Epoch {idx} Pass@1 {val_p_to_1}")
            f.write('\n')
            f.write(f"Train Epoch {idx} Pass@3 {val_p_to_3}")
            f.write('\n')
            f.write(f"Train Epoch {idx} Pass@5 {val_p_to_5}")
            f.write('\n')
            f.write(f"Train Epoch {idx} Pass@10 {val_p_to_10}")
            f.write('\n')
            f.write(f"Train Epoch {idx} Pass@15 {val_p_to_15}")
            f.write('\n')
            f.write(f"Validation Epoch {idx} MRR {eval_mrr}")
            f.write('\n')
            f.write(f"Validation Epoch {idx} Pass@1 {eval_p_to_1}")
            f.write('\n')
            f.write(f"Validation Epoch {idx} Pass@3 {eval_p_to_3}")
            f.write('\n')
            f.write(f"Validation Epoch {idx} Pass@5 {eval_p_to_5}")
            f.write('\n')
            f.write(f"Validation Epoch {idx} Pass@10 {eval_p_to_10}")
            f.write('\n')
            f.write(f"Validation Epoch {idx} Pass@15 {eval_p_to_15}")
            f.write('\n')
        if args.save_directory is not None:
            # Проверяем, существует ли уже директория
            if not os.path.exists(args.save_directory):
                os.makedirs(args.save_directory)
            torch.save(model.state_dict(), f"{args.save_directory}/ft_unixcoder_{idx}.bin")
        else:
            torch.save(model.state_dict(), f"ft_unixcoder_{idx}.bin") 
                       
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_dataset", default="df_train.pkl", type=str,
                        help="Датасет для обучения")
    parser.add_argument("--val_dataset", default="df_val.pkl", type=str,
                        help="Датасет для валидации")
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="Базовая модель, от которой обучаемся")
    parser.add_argument("--save_directory", default=None, type=str,
                        help="Директория, куда сохранять bin")
    parser.add_argument("--bin_name", default="", type=str,
                        help="Предобученные веса, если есть")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Имя токенайзера. Указывать, если не тот же самый, что и предобученной модели")
    parser.add_argument("--train_batch_size", default=50, type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--space_aug", action="store_true", default=False,
                        help="Нужно ли аугментировать код")
    parser.add_argument("--frequency_aug", action="store_true", default=False,
                        help="Будем ли делать анализ частотности")
    parser.add_argument("--frequency_all_text_aug", action="store_true", default=False,
                        help="Будем ли делать анализ частотности по всему тектсу (вариант Сергея)")
    parser.add_argument("--frequency_threshold", type=int, default=2,
                        help="Частотность, начиная с которой мы удаляем слова")
    parser.add_argument("--frequency_keep_first", action="store_true", default=False,
                        help="При использовании аугментации частотностью, удерживать ли первое вхождение")
    
    #print arguments
    args = parser.parse_args()

    loger_path = './logers/'

    if not os.path.exists('./logers/'):
        os.makedirs('logers')
    
    # создаем файловый обработчик логов
    file_handler = logging.FileHandler(os.path.join(loger_path, f'{date()}.log'))
    file_handler.setLevel(logging.DEBUG)
    
    # создаем форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # добавляем обработчик в логгер
    logger.addHandler(file_handler)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("device: %s",device)
    
    # Set seed
    set_seed(args.seed)

    #build model
    if args.tokenizer_name == "":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
    model.resize_token_embeddings(len(tokenizer))
    model = Model(model)

    if args.bin_name != "":
        checkpoint = torch.load(args.bin_name, map_location='cuda')
        model.load_state_dict(checkpoint)
    
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
            
    train(args, model, tokenizer)


if __name__ == "__main__":
    main()