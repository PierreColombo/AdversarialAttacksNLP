import os.path

import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import collate_fn
from sklearn.metrics import roc_auc_score
import json
import torch.nn.functional as F
from tqdm import tqdm
from utils_evaluation import *


def evaluate_adv(args, model, train_loader=None, clean_loader=None, adv_loader=None, path_to_save=None):
    """
    Signature is a bit different than OOD
    Args:
        args:
        model:
        train_samples:
        clean_samples:
        adv_samples:
        path_to_save:

    Returns:

    """

    def compute_scores(dataloader, id):
        current_outputs = []
        softmax_predictions = []
        energy_predictions = []
        inter = 0
        scores = {}
        l_labels = []
        for batch in tqdm(dataloader, 'Predictions for {}'.format(id)):
            inter += 1
            model.eval()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model.model(batch['input_ids'], attention_mask=batch['attention_mask'],
                                      output_hidden_states=True)
                logits = outputs['logits']
                if 'pooled' in list(outputs.keys()):
                    pooled = outputs['pooled']

                softmax_score = F.softmax(logits, dim=-1).detach().cpu().tolist()
                energy_scores = torch.logsumexp(logits, dim=-1).detach().cpu().tolist()
                softmax_predictions.append(softmax_score)
                energy_predictions.append(energy_scores)

                # Fait tout en même temps :
                # 1. Prendre les layers
                # 2. L1 aggregation
                # 3. Linf aggregation

                all_prediction = model.aggregeate(outputs)
                all_prediction['logits'] = logits.detach().cpu()
                l_labels.append(batch['labels'].detach().cpu().tolist())
                if 'pooled' in list(outputs.keys()):
                    all_prediction['pooled'] = pooled.detach().cpu()
                if args.use_all_layers:
                    for layer_number in range(len(outputs['hidden_states'])):
                        all_prediction['layer_{}'.format(layer_number)] = outputs['hidden_states'][layer_number][:, 0,
                                                                          :].detach().cpu()
                current_outputs.append(all_prediction)

        if len(scores) == 0:
            for key_input in all_prediction.keys():
                scores[key_input] = []
        for key_input in all_prediction.keys():
            if key_input != 'labels':  # TODO
                ood_keys = model.compute_adv(torch.cat([i[key_input] for i in current_outputs], dim=0),
                                             key_input)
                scores[key_input].append(ood_keys)

        softmax_predictions = sum(softmax_predictions, [])
        energy_predictions = sum(energy_predictions, [])
        final_scores = {}
        for k, v in scores.items():
            final_scores[k] = {}
            for sub_k in v[0].keys():
                list_v = []
                for i in range(len(v)):
                    list_v += v[i][sub_k]
                final_scores[k][sub_k] = list_v

        final_scores['softmax_predictions'] = softmax_predictions
        final_scores['energy_predictions'] = energy_predictions
        final_scores['labels'] = sum(l_labels, [])

        return final_scores

    if train_loader is not None:  # else we only comput ood score
        assert clean_loader is None
        assert adv_loader is None
        if not os.path.exists(path_to_save + 'train_score.json') and (
                not os.path.exists(path_to_save.replace('1', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('2', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('3', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('4', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('5', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('6', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('7', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('8', '0') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('9', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('8', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('7', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('6', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('5', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('4', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('3', '1') + 'train_score.json')) and (
                not os.path.exists(path_to_save.replace('2', '1') + 'train_score.json')):
            in_scores = compute_scores(train_loader, "train_score")

            with open(path_to_save + 'train_score.json', 'w') as file:
                json.dump(in_scores, file)

    else:
        assert clean_loader is not None
        assert adv_loader is not None
        assert train_loader is None
        if not os.path.exists(path_to_save + 'clean_score.json'):
            in_scores = compute_scores(clean_loader, "clean_score")

            with open(path_to_save + 'clean_score.json', 'w') as file:
                json.dump(in_scores, file)
        if not os.path.exists(path_to_save + 'adv_score.json'):
            out_scores = compute_scores(adv_loader, "adv_score")

            with open(path_to_save + 'adv_score.json', 'w') as file:
                json.dump(out_scores, file)

