#!/bin/bash

conf="examples/fastspeech2/conf/fastspeech2.kss.v2.yaml"
checkpoint="tf_models/fastspeech2.v2/kss/20210915/checkpoints/model-20000.h5"
devices="1"
batch_size="8"

. utils/parse_options.sh

inputdir=$1 # data_prep/ksss
outdir=$2   # tf_models/


echo "---------- MODEL CONFIG ----------"
echo -e "data dir: $data_dir\noutdir: $outdir\nconf: $conf"
echo -e "devices: $devices"
echo "----------------------------------"




# ------------------------------------------------
# Training
# ------------------------------------------------

CUDA_VISIBLE_DEVICES="$devices" python examples/fastspeech2/decode_fastspeech2.py \
  --rootdir $inputdir \
  --outdir $outdir \
  --config $conf \
  --checkpoint $checkpoint \
  --batch-size $batch_size