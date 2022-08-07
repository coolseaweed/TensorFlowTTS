#!/bin/bash


# This code is data prep for kss dataset

if [ ! -f data/archive.zip ];then
  echo "[ERROR] plz download kss data from kaggle: https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset " 
  exit 1
else
  echo "[INFO] kss data already downloaded!"
fi



if [ ! -d data/kss ]; then
  echo "[INFO] extract kss data"
  unzip data/archive.zip -d data/kss && mv data/kss/kss/1/1_0000.wav temp.wav && sox temp.wav -c 1 data/kss/kss/1/1_0000.wav && rm temp.wav
else
  echo "[INFO] skip extact kss data step"
fi 


if [ ! -d data_prep/kss ];then
  echo "[INFO] data prep."
  tensorflow-tts-preprocess --rootdir data/kss/ --outdir data_prep/kss --config preprocess/kss_preprocess.yaml --dataset kss
  tensorflow-tts-normalize --rootdir ./data_prep/kss --outdir ./data_prep/kss --config preprocess/kss_preprocess.yaml --dataset kss
  
else
  echo "[INFO] skip data prep step"
fi


if [ -f tf_models/tacotron2_v1/kss/model-100000.h5 ]; then
  
  steps="train valid"
  for step in $steps; do
    python examples/tacotron2/extract_duration.py \
        --rootdir ./data_prep/kss/${step} \
        --outdir ./data_prep/kss/${step}/durations \
        --checkpoint tf_models/tacotron2_v1/kss/model-100000.h5 \
        --use-norm 1 \
        --config examples/tacotron2/conf/tacotron2.kss.v1.yaml \
        --batch-size 32 \
        --win-front 3 \
        --win-back 3
    wait
  done

else
  echo "[ERROR] plz download tacotron2_v1 pretrained model from: https://drive.google.com/drive/folders/1WMBe01BBnYf3sOxMhbvnF2CUHaRTpBXJ" && exit 1
fi

echo "[INFO] successfully finished data prep!" && exit 0



