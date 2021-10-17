#!/bin/bash

conf="examples/fastspeech2/conf/fastspeech2.kss.v2.yaml"
conf_tacotron2="examples/tacotron2/conf/tacotron2.kss.v1.yaml"
tacotron2_model="tf_models/tacotron2.v1/kss/model-100000.h5"
model="mb-gan" # [fastspeech2, mb-gan]
devices="1"
resume=""

. utils/parse_options.sh

data_dir=$1 # data_prep/ksss
outdir=$2   # tf_models/


echo "---------- MODEL CONFIG ----------"
echo -e "data dir: $data_dir\noutdir: $outdir\nconf: $conf"
echo -e "devices: $devices"
echo "----------------------------------"



# ------------------------------------------------
# Data prep
# ------------------------------------------------
if [ ! -d $data_dir/train/durations ]; then

    echo "[INFO] $data_dir/train/durations directory doesn't exist! creating.."

    CUDA_VISIBLE_DEVICES="$devices" python examples/tacotron2/extract_duration.py \
    --rootdir "$data_dir/train" \
    --outdir "$data_dir/train/durations" \
    --checkpoint $tacotron2_model \
    --use-norm 1 \
    --config $conf_tacotron2 \
    --batch-size 32 \
    --win-front 3 \
    --win-back 3

fi

if [ ! -d $data_dir/valid/durations ]; then

    echo "[INFO] $data_dir/valid/durations directory doesn't exist! creating.."

    CUDA_VISIBLE_DEVICES="$devices" python examples/tacotron2/extract_duration.py \
    --rootdir "$data_dir/valid" \
    --outdir "$data_dir/valid/durations" \
    --checkpoint $tacotron2_model \
    --use-norm 1 \
    --config $conf_tacotron2 \
    --batch-size 32 \
    --win-front 3 \
    --win-back 3

fi


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