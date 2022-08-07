# 한국어 TTS 
## 실험환경
- GPU: 2080Ti x 1
- OS: ubuntu 18.04 (docker)
- python: 3.6.9
- tensorflow: 2.3.1

## 1. 환경 셋업
```bash
git clone -b r1.6.1 https://github.com/coolseaweed/TensorFlowTTS.git && git clone https://github.com/coolseaweed/utils.git TensorFlowTTS/utils
cd TensorFlowTTS && docker-compose up -d && docker exec -it tensorflowtts_tensorflowtts_1 bash
```

## 2. 데이터 준비
- KSS dataset [archive.zip](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset) 을 `data` directory에 옮기기
- tacotron2_v1 [pretrained model-100000.h5](https://drive.google.com/drive/folders/1WMBe01BBnYf3sOxMhbvnF2CUHaRTpBXJ) 을 `tf_models/tacotron2_v1/kss/` directory에 옮기기
```bash
./kss_data_prep.sh
```

## 3. Tensorflow TTS 학습
multi-band mel gan의 경우 200k iter 후 discriminator를 학습할 때 중단하게 되는데 (원인은 잘 모르겠다), `--resume` 커맨드와 함께 재시작하면 다시 학습하게된다.
```
# fastspeech2 (text -> mel-spectogram model)
./train.sh --model "fastspeech2" --conf examples/fastspeech2/conf/fastspeech2.kss.v2.yaml ./data_prep/kss tf_models/fastspeech2_v2/kss/

# multi-band mel gan (vocoder)
./train.sh --model "mb-gan" --conf examples/multiband_melgan/conf/multiband_melgan.v1.yaml ./data_prep/kss tf_models/mb_melgan_v1/kss/
./train.sh --model "mb-gan" --resume tf_models/mb_melgan_v1/kss/checkpoints/ckpt-200000 --conf examples/multiband_melgan/conf/multiband_melgan.v1.yaml ./data_prep/kss tf_models/mb_melgan_v1/kss/

```

## 4. Test
```
# inference
python inference.py --text2mel ./tf_models/fastspeech2_v2/kss/checkpoints/model-200000.h5 --text2mel_config ./tf_models/fastspeech2_v2/kss/config.yml --vocoder ./tf_models/mb_melgan_v1/kss/checkpoints/generator-200061.h5 --vocoder_config ./tf_models/mb_melgan_v1/kss/config.yml script.txt audio
```
