# 한국어 TTS 


## 1. 환경 셋업
```bash
git clone -b r1.6.1 https://github.com/coolseaweed/TensorFlowTTS.git
cd TensorFlowTTS && docker-compose up -d && docker exec -it tensorflowtts_tensorflowtts_1 bash
```

## 2. 데이터 준비
[kss dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset) 을 `data` directory에 옮기기
```bash
unzip data/archive.zip -d data/kss && mv data/kss/kss/1/1_0000.wav temp.wav && sox temp.wav -c 1 data/kss/kss/1/1_0000.wav && rm temp.wav
tensorflow-tts-preprocess --rootdir data/kss/ --outdir data_prep/kss --config preprocess/kss_preprocess.yaml --dataset kss
tensorflow-tts-normalize --rootdir ./data_prep/kss --outdir ./data_prep/kss --config preprocess/kss_preprocess.yaml --dataset kss

```

## 3. Tensorflow TTS 학습
```

```

## 4. Test

## 5. 모델최적화 + Tensorflow.js 모델 변환

