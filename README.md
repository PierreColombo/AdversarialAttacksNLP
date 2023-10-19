# Code for Toward Stronger Textual Attack Detectors

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* datasets
* wandb
* tqdm
* scikit-learn

## Dataset
We provide A SAMPLE OF Stakout in DATASET/ due to size constraints of openreview.

##  Repproducing experiements

The models we used are available in the HuggingFace Hub: https://huggingface.co/textattack. To generate the attacks we used TextAttack https://github.com/QData/TextAttack. 



### LAROUSSE & Mahanalobis


An example of bash example is given in:

```bash
>> sh eval_adv.sh
```

### GPT2 

```bash
>> python run_baseline_gpt2.py
```

### Multilayer Detectors

An example of bash example is given in:

```bash
>> sh get_distance.sh
```



