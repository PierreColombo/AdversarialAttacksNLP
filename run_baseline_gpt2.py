import json

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
import os
import pandas as pd

if __name__ == '__main__':

    # OPEN TEXT
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_ID = 'gpt2-large'
    print(f"Initializing {MODEL_ID}")
    gpt_model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
    gpt_model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)


    def compute_scores(texts):
        # ITTERATE THROUGH THE DS
        encodings = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True)

        batch_size = 1
        num_batch = len(texts) // batch_size
        likelihoods = []

        with torch.no_grad():
            for i in tqdm(range(num_batch), 'PPL'):
                start_idx = i * batch_size;
                end_idx = (i + 1) * batch_size
                x = encodings[start_idx:end_idx]
                ids = torch.LongTensor(x[0].ids)
                ids = ids.to(DEVICE)
                nll = gpt_model(input_ids=ids, labels=ids)[0]  # negative log-likelihood
                likelihoods.append(-1 * nll.item())
        logs = torch.tensor(likelihoods).tolist()
        return logs


    base_path = 'old_benchmark/cache_attack'
    use_old_benchmark = True
    path = base_path if use_old_benchmark else base_path + '_new'

    DS = [ 'sst2']  # 'ag-news', 'imdb',
    MODELS, SEEDS = ['bert-base-uncased', 'roberta-base'], range(10)
    ATTACKS = ['bae', 'pwws', 'textfooler', 'tf-adj'] if use_old_benchmark else ['deepwordbug', 'iga', 'kuleshov',
                                                                                 'clare', 'pruthi', 'textbugger']
    results = {}
    for ds in DS:
        for attack in tqdm(ATTACKS, 'attacks'):
            results[attack] = {}
            saving_path = f'result_gpt2_chineze_{attack}_{ds}.json' if use_old_benchmark else f'result_gpt2_extended_{attack}_{ds}.json'
            fsaving_path = os.path.join('results_baseline_gpt2', saving_path)
            # if not os.path.exists(fsaving_path):

            for model in tqdm(MODELS, 'models'):
                results[attack][model] = {}
                for seed in tqdm(SEEDS, 'seed'):
                    base_path = os.path.join(path, f'{model}-{ds}_{attack}_{seed}.csv')
                    if os.path.exists(base_path):
                        df_attacks = pd.read_csv(base_path)
                        text_in = df_attacks[df_attacks.result_type == 0]['text'].tolist()
                        text_out = df_attacks[df_attacks.result_type == 1]['text'].tolist()
                        results[attack][model][seed] = {
                            "in_scores": compute_scores(text_in),
                            "out_scores": compute_scores(text_out)
                        }
        with open(f'{ds}_{use_old_benchmark}', 'w') as file:
            json.dump(results, file)
