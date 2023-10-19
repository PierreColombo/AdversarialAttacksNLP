import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoModelForPreTraining, AutoModelForSequenceClassification
from sklearn.covariance import EmpiricalCovariance, MinCovDet, OAS, LedoitWolf
from data_depth import DataDepth
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA, PCA
from tqdm import tqdm


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = pooled = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, pooled


class CustomClassForAdversarialAttacks(nn.Module):
    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = model

    def aggregeate(self, outputs):
        if not self.args.do_not_aggregare_linf:
            return {
                'pw_1': sum(outputs['hidden_states'])[:, 0, :].detach().cpu(),
                'pw_inf': torch.cat(outputs['hidden_states'], dim=-1)[:, 0, :].detach().cpu()
            }
        else:
            return {
                'pw_1': sum(outputs['hidden_states'])[:, 0, :].detach().cpu(),

            }

    def compute_adv(
            self, layer_output_unreduce, string_id
    ):
        layer_output = layer_output_unreduce
        if self.args.use_reduction:
            layer_output = torch.tensor(
                self.reduced_estimator[string_id].transform(layer_output_unreduce.detach().cpu().numpy()))
        with torch.no_grad():
            ood_keys = {}

            maha_score = {}
            # TODO : faire les bails ici :)
            use_estimator = True

            for k_estimator in list(self.dic_estimators.keys()):
                maha_score[k_estimator] = []

                if self.args.use_only_one_depth:
                    ms = torch.tensor(
                        self.class_estimator[k_estimator][string_id].mahalanobis(layer_output.cpu()))  # TODO
                    maha_score[k_estimator] = ms.tolist()
                else:
                    for c in tqdm(self.all_classes, 'Classes {}'.format('Mahalonobist')):
                        if self.class_estimator[k_estimator][string_id][c] is not None:
                            ms = torch.tensor(
                                self.class_estimator[k_estimator][string_id][c].mahalanobis(layer_output.cpu()))
                            maha_score[k_estimator].append(ms)
                        else:
                            use_estimator = False

                    if use_estimator:
                        maha_score[k_estimator] = torch.stack(maha_score[k_estimator], dim=-1)
                    else:
                        del maha_score[k_estimator]
            depth_scores_all = {}
            if not self.args.do_not_compute_depths:
                depth = DataDepth(10000)

                for depth_name in tqdm(["halfspace_mass"],
                                       # "int_w_halfs_pace", "halfspace_mass" "half_space", "proj_depth",
                                       'depth choice'):
                    print('Computing depths', depth_name)
                    depth_scores_all['{}_unn'.format(depth_name)] = []

                    if self.args.use_only_one_depth:
                        x_train = self.bank[string_id].detach().cpu().numpy()
                        x_test = layer_output.cpu().numpy()
                        ms = depth.compute_depths(np.array(x_train, dtype=np.float64),
                                                  np.array(x_test, dtype=np.float64), depth_name)
                        depth_scores_all['{}_single_unn'.format(depth_name)] = ms.tolist()

                    else:

                        for c in tqdm(self.all_classes, 'Classes {}'.format(depth_name)):
                            x_train = self.bank[string_id][self.label_bank == c].detach().cpu().numpy()
                            x_test = layer_output.cpu().numpy()
                            try:
                                ms = depth.compute_depths(np.array(x_train, dtype=np.float64),
                                                          np.array(x_test, dtype=np.float64), depth_name)
                            except:
                                print('Error while computing depths {} for class {}'.format(depth_name, c))
                                ms = np.zeros((np.array(x_test, dtype=np.float64).shape[0]))
                            depth_scores_all['{}_unn'.format(depth_name)].append(ms)
                        depth_scores_all['{}_unn'.format(depth_name)] = np.stack(
                            depth_scores_all['{}_unn'.format(depth_name)])
                        depth_scores_all['{}_unn'.format(depth_name)] = torch.tensor(
                            depth_scores_all['{}_unn'.format(depth_name)]).tolist()

            norm_pooled = F.normalize(layer_output, dim=-1)
            cosine_score = norm_pooled.cpu() @ self.norm_bank[string_id].t().cpu()
            cosine_score = cosine_score.max(-1)[0]

            ood_keys['cosine'] = cosine_score.tolist()
            # ood_keys['maha'] = maha_score.tolist()
            for key, value in maha_score.items():
                ood_keys[key] = torch.tensor(value).tolist()
            for key, value in depth_scores_all.items():
                try:
                    ood_keys[key] = torch.tensor(value).permute(1, 0).tolist()
                except:
                    ood_keys[key] = torch.tensor(value).tolist()
        return ood_keys

    def prepare_adv(self, dataloader=None):
        self.bank = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_bank = None
        iterator = 0
        all_predictions = []
        for batch in tqdm(dataloader, "Preparing For Mahanalobist"):
            iterator += 1
            self.eval()
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch['labels']
            with torch.no_grad():
                outputs = self.model(batch['input_ids'],
                                     attention_mask=batch['attention_mask'], output_hidden_states=True
                                     )  # TODO :faire les bails ici
                all_prediction = self.aggregeate(outputs)
                all_prediction['logits'] = outputs['logits'].detach().cpu()
                if 'pooled' in list(outputs.keys()):
                    all_prediction['pooled'] = outputs['pooled'].detach().cpu()
                all_prediction['labels'] = labels.clone().detach().cpu()
                if self.args.use_all_layers:
                    for layer_number in range(len(outputs['hidden_states'])):
                        all_prediction['layer_{}'.format(layer_number)] = outputs['hidden_states'][layer_number][:, 0,
                                                                          :].detach().cpu()
                all_predictions.append(all_prediction)

        self.bank = {}
        self.norm_bank = {}
        self.class_mean = {}
        self.class_var = {}

        self.reduced_estimator = {}
        self.reduced_bank = {}

        for k in all_prediction.keys():  # iterrate through the layers/outputs/etc.
            print('Itterating through', k)
            if 'label' not in k:
                bank = torch.cat([i[k] for i in all_predictions], dim=0)
                if self.args.use_reduction:
                    estimator = KernelPCA(n_components=self.args.dim_kernel, gamma=(1 / bank.shape[-1]), \
                                          kernel='rbf', random_state=10)
                    bank = torch.tensor(estimator.fit_transform(bank.detach().cpu().numpy()))
                    self.reduced_estimator[k] = estimator
                self.bank[k] = bank
                self.norm_bank[k] = F.normalize(bank, dim=-1)
            else:
                self.label_bank = torch.cat([i[k] for i in all_predictions])
                self.all_classes = list(set(self.label_bank.tolist()))

        self.dic_estimators = {
            "empirical": EmpiricalCovariance,
            "OAS": OAS,
            # "LedoitWolf": LedoitWolf,
            # "MinCovDet": MinCovDet

        }

        self.class_estimator = {}
        for type_of_estimator, estimator in tqdm(self.dic_estimators.items(), 'Fitting Estimators'):
            self.class_estimator[type_of_estimator] = {}
            for k in tqdm(all_prediction.keys(), 'Layers'):
                self.class_estimator[type_of_estimator][k] = {}
                try:
                    if 'label' not in k:
                        N, d = self.norm_bank[k].size()
                        # self.class_mean[k] = torch.zeros(max(self.all_classes) + 1, d).detach().cpu()
                        if self.args.use_only_one_depth:
                            print('Fitting for single')
                            self.class_estimator[type_of_estimator][k] = estimator().fit(
                                self.bank[k].detach().cpu())
                            print('Fitted success for')
                        else:
                            for c in self.all_classes:
                                print('Fitting for', c)
                                self.class_estimator[type_of_estimator][k][c] = estimator().fit(
                                    self.bank[k][self.label_bank == c].detach().cpu())
                                print('Fitted success for', c)
                except:
                    if self.args.use_only_one_depth:
                        self.class_estimator[type_of_estimator][k] = None
                    else:
                        for c in self.all_classes:
                            self.class_estimator[type_of_estimator][k][c] = None
