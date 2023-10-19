export HF_DATASETS_CACHE="datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


for SEED in 0 1 2 3 4 5 6; do
  for MODEL in 'bert-base-uncased' 'roberta-base'; do
    for DS in 'imdb' "sst2" 'ag-news'; do
      for ATTACK in 'deepwordbug' 'bae' 'pwws' 'pruthi' 'textbugger' 'iga' 'deepwordbug' 'kuleshov' 'clare' 'checklist' 'input-reduction'; do
        export SUFFIX=${MODEL}_${DS}_${ATTACK}
        echo $SUFFIX
        sbatch --job-name=${SUFFIX} \
          --gres=gpu:3 \
          --no-requeue \
          --cpus-per-task=30 \
          --hint=nomultithread \
          --time=5:00:00 \
          -C v100-32g \
          --output=jobinfo/${SUFFIX}_%j.out \
          --error=jobinfo/${SUFFIX}_%j.err \
          --wrap="module purge; module load pytorch-gpu/py3/1.9.0 ; export HF_DATASETS_CACHE="/datasets"; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1 ;  python eval_pretrained_model_attack.py --use_old_benchmark_extended --seed=$SEED  --do_not_aggregare_linf --model=$MODEL --dataset=$DS --use_all_layers --suffix_name=$SUFFIX --attack_type=$ATTACK --batch_size=256 "

      done
    done
  done
done