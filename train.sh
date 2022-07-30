#!/bin/bash

conf="examples/fastspeech2/conf/fastspeech2.kss.v2.yaml"
model="mb-gan" # [fastspeech2, mb-gan]
devices="1"
resume=""   # checkpoint path 

. utils/parse_options.sh

data_dir=$1 # data_prep/kss
outdir=$2   # tf_models/


echo "---------- MODEL CONFIG ----------"
echo -e "data dir: $data_dir\noutdir: $outdir\nconf: $conf"
echo -e "devices: $devices"
echo "----------------------------------"


# ------------------------------------------------
# Training
# ------------------------------------------------

if [[ $model == "fastspeech2" ]]; then
    CUDA_VISIBLE_DEVICES="$devices" \
    python examples/fastspeech2/train_fastspeech2.py \
        --train-dir "$data_dir/train/" \
        --dev-dir "$data_dir/valid/" \
        --outdir $outdir \
        --config $conf \
        --use-norm 1 \
        --f0-stat $data_dir/stats_f0.npy \
        --energy-stat $data_dir/stats_energy.npy \
        --mixed_precision 1 \
        --resume "$resume"

elif [[ $model == "mb-gan" ]]; then
    CUDA_VISIBLE_DEVICES="$devices" \
    python examples/multiband_melgan/train_multiband_melgan.py \
        --train-dir "$data_dir/train/" \
        --dev-dir "$data_dir/valid/" \
        --outdir $outdir \
        --config $conf \
        --use-norm 1 \
        --generator_mixed_precision 1 \
        --resume "$resume"
fi


echo "[INFO] training $model model finished!" && exit 0
