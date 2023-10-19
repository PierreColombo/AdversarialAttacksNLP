import argparse
import json
import os
import torch
from data import load, get_labels_zh
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from utils import set_seed, collate_fn_adv
import ot
from model import CustomClassForAdversarialAttacks
from evaluation import evaluate_adv
import warnings
from data import load
import sys
import random
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from eval_pretrained_model_attack import get_dasetes_for_adv, get_clean_samples, MODEL_TO_MODEL

if __name__ == '__main__':
    MODELS = ["roberta-base", "bert-base-uncased"]
    DATASETS = ['sst2', 'rte', 'ag-news', 'imdb']

    USE_OLD_BENCHMARK = True
    USE_OLD_BENCHMARK_EXTENDED = False
    layers, clean_adv, train_clean, train_adv, seeds, attacks, dss, models = [], [], [], [], [], [], [], []
    SEEDS = range(4)
    nb_layers = 13
    ATTACKS = ['bae', 'pwws', 'pruthi', 'textbugger', 'iga', 'deepwordbug', 'kuleshov',
               'input-reduction', 'clare', 'checklist', 'textfooler', 'tf-adj']
    print('Starting The loops')
    os.makedirs('get_distances_results', exist_ok=True)
    for selected_model in tqdm(MODELS, 'models'):
        for selected_dataset in tqdm(DATASETS, 'ds'):
            print('Loading Model')
            model_name = MODEL_TO_MODEL[selected_dataset][selected_model]
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            for selected_attack in tqdm(ATTACKS, 'attack'):
                for selected_seed in tqdm(SEEDS, 'seed'):

                    model.eval()

                    if USE_OLD_BENCHMARK:
                        path_to_log_csv = os.path.join('old_benchmark', 'cache_attack',
                                                       '{}-{}_{}_{}.csv'.format(selected_model, selected_dataset,
                                                                                selected_attack, selected_seed))
                    elif USE_OLD_BENCHMARK_EXTENDED:
                        path_to_log_csv = os.path.join('old_benchmark', 'cache_attack_new',
                                                       '{}-{}_{}_{}.csv'.format(selected_model, selected_dataset,
                                                                                selected_attack, selected_seed))

                    else:
                        raise NotImplementedError
                    if os.path.exists(path_to_log_csv):
                        data_dic, num_labels = get_dasetes_for_adv(selected_dataset, tokenizer, path_to_log_csv,
                                                                   USE_OLD_BENCHMARK, USE_OLD_BENCHMARK_EXTENDED)
                        train_dataset = []
                        for index, result in enumerate(data_dic['train_samples']):
                            if index == 2000:
                                break
                            result['labels'] = data_dic['train_labels'][index]
                            train_dataset.append(result)

                        z_train, y_train = [], []
                        for index, train_sample in tqdm(enumerate(train_dataset), 'Train'):
                            y_train.append(train_sample['labels'])
                            z_train.append(model(input_ids=train_sample['input_ids'],
                                                 attention_mask=train_sample['attention_mask'],
                                                 output_hidden_states=True)['hidden_states'])

                        adv_dataset = []
                        for index, result in enumerate(data_dic['adv_samples_tok']):
                            result['labels'] = data_dic['adv_label'][index]
                            adv_dataset.append(result)

                        clean_dataset = []
                        for index, result in enumerate(data_dic['clean_samples_tok']):
                            result['labels'] = data_dic['clean_label'][index]
                            clean_dataset.append(result)

                        z_clean, y_clean = [], []
                        for index, clean_sample in tqdm(enumerate(clean_dataset), 'Clean'):
                            y_clean.append(clean_sample['labels'])
                            z_clean.append(model(input_ids=clean_sample['input_ids'],
                                                 attention_mask=clean_sample['attention_mask'],
                                                 output_hidden_states=True)['hidden_states'])

                        z_adv, y_adv = [], []
                        for index, adv_sample in tqdm(enumerate(adv_dataset), 'Adv'):
                            y_adv.append(adv_sample['labels'])
                            z_adv.append(model(input_ids=adv_sample['input_ids'],
                                               attention_mask=adv_sample['attention_mask'],
                                               output_hidden_states=True)['hidden_states'])

                        for layer_to_select in range(nb_layers):
                            print(layer_to_select)
                            a = np.array([1 / len(z_clean) for _ in range(len(z_clean))])
                            b = np.array([1 / len(z_adv) for _ in range(len(z_adv))])
                            x, y = np.array([i[layer_to_select].detach().squeeze(0)[
                                                 0].numpy() for i in z_clean]), np.array(
                                [i[layer_to_select].detach().squeeze(0)[
                                     0].numpy() for i in z_adv])
                            M = euclidean_distances(x, y)
                            ot_distance = ot.emd2(a, b, M)

                            layers.append(layer_to_select)
                            seeds.append(selected_seed)
                            attacks.append(selected_attack)
                            models.append(selected_model)
                            dss.append(selected_dataset)

                            clean_adv.append(ot_distance)
                            a = np.array([1 / len(z_train) for _ in range(len(z_train))])
                            b = np.array([1 / len(z_adv) for _ in range(len(z_adv))])
                            x, y = np.array([i[layer_to_select].detach().squeeze(0)[
                                                 0].numpy() for i in z_train]), np.array(
                                [i[layer_to_select].detach().squeeze(0)[
                                     0].numpy() for i in z_adv])
                            M = euclidean_distances(x, y)
                            ot_distance = ot.emd2(a, b, M)
                            train_clean.append(ot_distance)
                            a = np.array([1 / len(z_clean) for _ in range(len(z_clean))])
                            b = np.array([1 / len(z_train) for _ in range(len(z_train))])
                            x, y = np.array([i[layer_to_select].detach().squeeze(0)[
                                                 0].numpy() for i in z_clean]), np.array(
                                [i[layer_to_select].detach().squeeze(0)[
                                     0].numpy() for i in z_train])
                            M = euclidean_distances(x, y)
                            ot_distance = ot.emd2(a, b, M)
                            train_adv.append(ot_distance)

                final_dic = {
                    'layers': layers,
                    'seeds': seeds,
                    'clean_adv': clean_adv,
                    'train_clean': train_clean,
                    'train_adv': train_adv,
                    'dss': dss,
                    'models': models,
                    'attacks': attacks,
                }
                print(final_dic)
                with open(
                        os.path.join('get_distances_results',
                                     f'get_distance_{selected_dataset}_{selected_model}_{selected_attack}_{USE_OLD_BENCHMARK}_{USE_OLD_BENCHMARK_EXTENDED}.json'),
                        'w') as file:
                    json.dump(final_dic, file)
                layers, clean_adv, train_clean, train_adv, seeds, attacks, dss, models = [], [], [], [], [], [], [], []
