# 한국어 TTS 
[한국어 TTS demo](https://goarcade.net/tts/)


## Env. setup 

NVIDIA-driver version 에 따라 tensorflow-gpu 에 맞는 버전 골라서 셋업  

- export user id
    ```bash
    # add following commands on ~/.bashrc or just type once
    export USER_ID=$(id -u) 
    export GROUP_ID=$(id -g)
    ```
- TensorflowTTS

    ```bash
    docker-compose -f docker-compose-tf{2.6,2.3}.yaml up -d
    ```
- TensorflowTTS-js
    ```bash
    docker-compose -f docker-compose-tfjs{2.6,2.3}.yaml up -d
    ```

---
## Data prep.
- KSS dataset [archive.zip](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset) 을 `data` directory에 옮기기
- tacotron2_v1 [pretrained model-100000.h5](https://drive.google.com/drive/folders/1WMBe01BBnYf3sOxMhbvnF2CUHaRTpBXJ) 을 `tf_models/tacotron2_v1/kss/` directory에 옮기기
    ```bash
    ./kss_data_prep.sh
    ```
---
## Training
multi-band mel gan의 경우 200k iter 후 discriminator를 학습할 때 중단하게 되는데, `--resume` 커맨드와 함께 재시작하면 다시 학습하게된다.
```
# fastspeech2 (text2mel-spectogram model)
./train.sh \
    --model "fastspeech2" \
    --conf examples/fastspeech2/conf/fastspeech2.kss.v2.yaml \
    ./data_prep/kss tf_models/fastspeech2_v2/kss/

# multi-band mel gan (vocoder)
./train.sh \
    --model "mb-gan" \
    --conf examples/multiband_melgan/conf/multiband_melgan.v1.yaml \
    ./data_prep/kss tf_models/mb_melgan_v1/kss/

./train.sh \
    --model "mb-gan" \
    --resume tf_models/mb_melgan_v1/kss/checkpoints/ckpt-200000 \
    --conf examples/multiband_melgan/conf/multiband_melgan.v1.yaml \
    ./data_prep/kss tf_models/mb_melgan_v1/kss/

```
---
## Inference
```
# inference
python inference.py \
    --text2mel ./tf_models/fastspeech2_v2/kss/checkpoints/model-200000.h5 \
    --text2mel_config ./tf_models/fastspeech2_v2/kss/config.yml \
    --vocoder ./tf_models/mb_melgan_v1/kss/checkpoints/generator-200061.h5 \
    --vocoder_config ./tf_models/mb_melgan_v1/kss/config.yml \
    <input text file> <output audio directroy>
```
---
## Keras2tfSaved
Tensorflow js 로 exporting 하기전에 saved file로 변환

input model directory 에는 `*.h5` `*.yaml` 확장자의 모델파일과 config 파일 필요


```
# input model tree
models/
├── config.yml
├── generator-375000.h5

```

```
python keras2saved.py \
    --input <input model dir> \
    --output <output model dir>

```
---
## Tensorflow-js
[link](https://github.com/coolseaweed/TensorFlowTTS/tree/prod/tensorflow_js)

---
## LICENSE
All code is MIT licensed.
