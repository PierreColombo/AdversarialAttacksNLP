for SEED in 0; do
  for MODEL in 'bert-base-uncased'; do
    for DS in 'imdb' "sst2" 'ag-news'; do
      for ATTACK in 'deepwordbug' 'bae' 'pwws' 'pruthi' 'textbugger' 'iga' 'deepwordbug' 'kuleshov' 'clare' 'checklist' 'input-reduction'; do
        export SUFFIX=${MODEL}_${DS}_${ATTACK}
        echo $SUFFIX
        sbatch --job-name=${SUFFIX} \
          --account= \
          --gres=gpu:3 \
          --no-requeue \
          --cpus-per-task=30 \
          --hint=nomultithread \
          --time=20:00:00 \
          --output=jobinfo/${SUFFIX}_%j.out \
          --error=jobinfo/${SUFFIX}_%j.err \
          --wrap="module purge; module load pytorch-gpu/py3/1.9.0 ; export HF_DATASETS_CACHE="datasets"; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1 ;  python get_distances.py"
      done
    done
  done
done
