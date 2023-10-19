import argparse
import os
import torch
from data import load, get_labels_zh
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from utils import set_seed, collate_fn_adv

from model import CustomClassForAdversarialAttacks
from evaluation import evaluate_adv
import warnings
from data import load
import sys
import random
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import pandas as pd




def get_dasetes_for_adv(dataset, tokenizer, path_to_log_csv, use_old_benchmark, use_old_benchmark_extended):
    # For NLI dataset, only get the hypothesis, which is attacked what about RTE ?
    if dataset == "ag-news":
        train_dataset = load_dataset("ag_news")['train'].shard(num_shards=5, index=0)  # text
        keys = ['text']
    elif dataset == "wnli":
        train_dataset = load_dataset("glue", "wnli")['train']  # sentence1 sentence2
        keys = ['sentence1', 'sentence2']
    elif dataset == "rte":
        train_dataset = load_dataset("glue", "rte")['train']  # sentence1 sentence2
        keys = ['sentence1', 'sentence2']
    elif dataset == "sst2":
        train_dataset = load_dataset("glue", "sst2")[
            'train']  # .shard(num_shards=5, index=0) # TODO : avant  # sentence
        keys = ['sentence']
    elif dataset == "imdb":
        train_dataset = load_from_disk('imdb_train')
        keys = ['text']
    else:
        raise NotImplementedError
    num_labels = len(set(train_dataset['label']))
    # train_dataset = dataset.shuffle(seed=0)

    train_samples = []
    train_labels = []

    def clean(string):
        for i in range(20):
            string = string.replace('[', '').replace(']', '')
        return string

    for index, sample in tqdm(enumerate(train_dataset)):
        train_samples.append(
            tokenizer(' '.join([clean(sample[key]) for key in keys]), return_tensors="pt", truncation=True))
        train_labels.append(sample["label"])

    #### OPENING LOGS OF GENERATED ATTACK ####
    df_attacks = pd.read_csv(path_to_log_csv)

    if (not use_old_benchmark and not use_old_benchmark_extended):
        #### CLEAN SCORE ####
        clean_samples = df_attacks['original_text'].tolist()
        clean_label = df_attacks['original_output'].tolist()
        adv_samples = df_attacks[df_attacks.result_type == 'Successful']['perturbed_text'].tolist()
        adv_label = df_attacks[df_attacks.result_type == 'Successful']['perturbed_output'].tolist()

    else:
        clean_samples = df_attacks[df_attacks.result_type == 0]['text'].tolist()
        clean_label = df_attacks[df_attacks.result_type == 0]['original_output'].tolist()
        adv_samples = df_attacks[df_attacks.result_type == 1]['text'].tolist()
        adv_label = df_attacks[df_attacks.result_type == 1]['perturbed_output'].tolist()

    clean_samples_tok = []
    for index, sample in enumerate(clean_samples):
        for _ in range(20):
            sample = sample.replace('[', '').replace(']', '')
        clean_samples_tok.append(tokenizer(sample, return_tensors="pt", truncation=True))

    #### ADV SCORE ####

    adv_samples_tok = []
    for index, sample in enumerate(adv_samples):
        for _ in range(20):
            sample = sample.replace('[', '').replace(']', '')
        adv_samples_tok.append(tokenizer(sample, return_tensors="pt", truncation=True))

    return {
               "clean_samples_tok": clean_samples_tok,
               "clean_label": clean_label,
               "adv_samples_tok": adv_samples_tok,
               "adv_label": adv_label,
               "train_samples": train_samples,
               "train_labels": train_labels
           }, num_labels


def get_clean_samples(adv_samples_tok, adv_labels, clean_samples_tok, clean_labels, seed):
    random.seed(seed)
    length = min(len(adv_labels), len(clean_labels))
    clean_samples_tok_sub, clean_labels_sub = zip(*random.sample(list(zip(clean_samples_tok, clean_labels)), length))
    adv_samples_tok_sub, adv_labels_sub = zip(*random.sample(list(zip(adv_samples_tok, adv_labels)), length))
    return adv_samples_tok_sub, adv_labels_sub, clean_samples_tok_sub, clean_labels_sub


MODEL_TO_MODEL = {
    "imdb": {
        "roberta-base": "textattack/roberta-base-imdb",
        "albert-base-v2": "textattack/albert-base-v2-imdb",
        "bert-base-uncased": "textattack/bert-base-uncased-imdb",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-imdb"
    },
    "sst2": {
        "roberta-base": "textattack/roberta-base-SST-2",
        "albert-base-v2": "textattack/albert-base-v2-SST-2",
        "bert-base-uncased": "textattack/bert-base-uncased-SST-2",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-SST-2"
    },
    "ag-news": {
        "roberta-base": "textattack/roberta-base-ag-news",
        "albert-base-v2": "textattack/albert-base-v2-ag-news",
        "bert-base-uncased": "textattack/bert-base-uncased-ag-news",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-ag-news"
    },
    "wnli": {
        "roberta-base": "textattack/roberta-base-WNLI",
        "albert-base-v2": "textattack/albert-base-v2-WNLI",
        "bert-base-uncased": "textattack/bert-base-uncased-WNLI",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-WNLI"
    },
    "rte": {
        "roberta-base": "textattack/roberta-base-RTE",
        "albert-base-v2": "textattack/albert-base-v2-RTE",
        "bert-base-uncased": "textattack/bert-base-uncased-RTE",
        "distilbert-base-uncased": "textattack/distilbert-base-uncased-RTE"
    },

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased",
                        choices=["distilbert-base-uncased", "roberta-base", "albert-base-v2", "bert-base-uncased"],
                        type=str)
    parser.add_argument("--dataset", default="ag-news",
                        choices=['sst2', "wnli", 'rte', 'ag-news', 'imdb'],
                        type=str)
    parser.add_argument("--use_old_benchmark", action='store_true')
    parser.add_argument("--use_old_benchmark_extended", action='store_true')
    parser.add_argument("--use_only_one_depth", action='store_true')
    parser.add_argument("--use_all_layers", action='store_true')
    parser.add_argument("--use_reduction", action='store_true')
    parser.add_argument("--do_not_compute_depths", action='store_true')
    parser.add_argument("--do_not_aggregare_linf", action='store_true')
    parser.add_argument("--number_of_seeds", default=10, type=int)
    parser.add_argument("--dim_kernel", default=100, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--suffix_name", type=str, default='debug')
    parser.add_argument("--attack_type", default="iga", type=str,
                        choices=['bae', 'pwws', 'pruthi', 'textbugger', 'iga', 'deepwordbug', 'kuleshov',
                                 'input-reduction', 'clare', 'checklist', 'textfooler', 'tf-adj'])
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()
    assert args.seed in list(range(10))
    print(args)

    uuid = "{}_{}_{}_{}".format(args.model, args.dataset, args.attack_type, args.seed)
    if args.suffix_name is not None:
        uuid += '_{}'.format(args.suffix_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args)

    model_name = MODEL_TO_MODEL[args.dataset][args.model]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = CustomClassForAdversarialAttacks(args, tokenizer, model)
    model.to(device)
    model.eval()

    # GET TRAINSET
    print('Train Set')
    path_to_save_train = 'adver_log_train_{}_{}_{}_{}_{}_{}'.format(args.use_reduction,
                                                                             args.use_old_benchmark,
                                                                             args.use_only_one_depth,
                                                                             args.use_old_benchmark_extended,
                                                                             args.do_not_compute_depths,
                                                                             args.do_not_aggregare_linf)

    os.makedirs(path_to_save_train, exist_ok=True)
    path_to_save = os.path.join(path_to_save_train, uuid)
    if args.use_old_benchmark:
        path_to_log_csv = os.path.join('old_benchmark', 'cache_attack',
                                       '{}-{}_{}_{}.csv'.format(args.model, args.dataset, args.attack_type, args.seed))
    elif args.use_old_benchmark_extended:
        path_to_log_csv = os.path.join('old_benchmark', 'cache_attack_new',
                                       '{}-{}_{}_{}.csv'.format(args.model, args.dataset, args.attack_type, args.seed))
    else:
        path_to_log_csv = os.path.join('generate_attacks', args.dataset, args.model, args.attack_type, 'logs.csv')
    data_dic, num_labels = get_dasetes_for_adv(args.dataset, tokenizer, path_to_log_csv, args.use_old_benchmark,
                                               args.use_old_benchmark_extended)

    train_dataset = []
    for index, result in enumerate(data_dic['train_samples']):
        result['labels'] = data_dic['train_labels'][index]
        train_dataset.append(result)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_adv)
    with torch.no_grad():
        model.prepare_adv(train_dataloader)  # TODO à faire :)
    print('Eval Train Set')

    evaluate_adv(args, model, train_dataloader, None, None, path_to_save=path_to_save)  # TODO à faire :)
    path_to_save_attacks = 'adver_log_attack_{}_{}_{}_{}_{}_{}'.format(args.use_reduction,
                                                                                args.use_old_benchmark,
                                                                                args.use_only_one_depth,
                                                                                args.use_old_benchmark_extended,
                                                                                args.do_not_compute_depths,
                                                                                args.do_not_aggregare_linf)
    # False_False_True_False_True
    os.makedirs(path_to_save_attacks, exist_ok=True)
    print('Test Set')
    for seed in tqdm(range(args.number_of_seeds), 'seeds'):

        path_to_save_seed = os.path.join(path_to_save_attacks,
                                         "{}_{}_{}_{}".format(args.model, args.dataset, args.attack_type, seed))
        adv_samples_tok_sub, adv_labels_sub, clean_samples_tok_sub, clean_labels_sub = get_clean_samples(
            data_dic["adv_samples_tok"],
            data_dic["adv_label"],
            data_dic["clean_samples_tok"],
            data_dic["clean_label"], seed)
    path_to_save_seed = os.path.join(path_to_save_attacks,
                                     "{}_{}_{}_{}".format(args.model, args.dataset, args.attack_type, args.seed))
    adv_dataset = []
    for index, result in enumerate(data_dic['adv_samples_tok']):
        result['labels'] = data_dic['adv_label'][index]
        adv_dataset.append(result)
    adv_dataloader = DataLoader(adv_dataset, batch_size=args.batch_size, collate_fn=collate_fn_adv)

    clean_dataset = []
    for index, result in enumerate(data_dic['clean_samples_tok']):
        result['labels'] = data_dic['clean_label'][index]
        clean_dataset.append(result)
    clean_dataloader = DataLoader(clean_dataset, batch_size=args.batch_size, collate_fn=collate_fn_adv)

    evaluate_adv(args, model, None, clean_dataloader, adv_dataloader, path_to_save_seed)
