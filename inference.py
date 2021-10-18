import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig



def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
  input_ids = processor.text_to_sequence(input_text)


  # text2mel part
  if text2mel_name == "TACOTRON":
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
  elif text2mel_name == "FASTSPEECH2":
    mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  else:
    raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")


  # vocoder part
  if vocoder_name == "MB-MELGAN":
    audio = vocoder_model.inference(mel_outputs)[0, :, 0]
  else:
    raise ValueError("Only MB_MELGAN are supported on vocoder_name")

  if text2mel_name == "TACOTRON":
    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
  else:
    return mel_outputs.numpy(), audio.numpy()



# initialize fastspeech2 model
fastspeech2_config = AutoConfig.from_pretrained('tf_models/fastspeech2.v2/kss/20210915/config.yml')
fastspeech2 = TFAutoModel.from_pretrained(
    config=fastspeech2_config,
    pretrained_path="tf_models/fastspeech2.v2/kss/baseline/model-200000.h5",
    name="fastspeech2"
)

# initialize melgan model
mb_melgan_config = AutoConfig.from_pretrained('tf_models/mb_melgan.v1/kss/20210916/config.yml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    pretrained_path="tf_models/mb_melgan.v1/kss/baseline/generator-1000000.h5",
    name="mb_melgan"
)

processor = AutoProcessor.from_pretrained(pretrained_path="./test/files/kss_mapper.json")

FILE='script.txt'
with open(FILE, 'r') as f:
  for i, line in enumerate(f):
    input_text = line.strip()
    _, audio = do_synthesis(input_text, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN")
    sf.write(f'./audio/line{i}.wav', audio, 22050, "PCM_16")
    print(i, input_text)

