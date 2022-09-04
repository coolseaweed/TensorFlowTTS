import numpy as np
import soundfile as sf
import yaml
import argparse
import tensorflow as tf
import sys
import os
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--text2mel-path',
        dest="text2mel_path",
        default="models/TF/FASTSPEECH2/model-200000.h5",
        help='text2mel model path'
    )

    parser.add_argument(
        '--text2mel-config',
        dest="text2mel_config",
        default="models/TF/FASTSPEECH2/config.yaml",
        help='text2mel config path'
    )

    parser.add_argument(
        '--text2mel-name',
        dest="text2mel_name",
        default="FASTSPEECH2",
        help='text2mel config path'
    )

    parser.add_argument(
        '--vocoder-path',
        dest="vocoder_path",
        default="models/TF/MB_MELGAN_HF/generator-375000.h5",
        help='vocoder model path'
    )

    parser.add_argument(
        '--vocoder-config',
        dest="vocoder_config",
        default="models/TF/MB_MELGAN_HF/config.yaml",
        help='text2mel config path'
    )
    parser.add_argument(
        '--vocoder-name',
        dest="vocoder_name",
        default="MB_MELGAN_HF",
        help='text2mel config path'
    )

    parser.add_argument(
        'input',
        default='input.txt',
        help='input text file'
    )

    parser.add_argument(
        'output',
        default='output',
        help='out audio dir'
    )

    args = parser.parse_args()

    return args


def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name, processor):
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

    print(f"mel output: {mel_outputs.shape} / type: {type(mel_outputs)}")
    # vocoder part
    if vocoder_name == "MB_MELGAN":
        audio = vocoder_model.inference(mel_outputs)[0, :, 0]
    elif vocoder_name == "HIFI_GAN":
        audio = vocoder_model.inference(mel_outputs)[0, :, 0]
    elif vocoder_name == "MB_MELGAN_HF":
        audio = vocoder_model.inference(mel_outputs)[0, :, 0]
    else:
        raise ValueError("Only MB_MELGAN are supported on vocoder_name")

    if text2mel_name == "TACOTRON":
        return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
    else:
        return mel_outputs.numpy(), audio.numpy()


def check_params(arg):
    arg_dict = vars(arg)
    print(" ------- PARAMS ------- ")
    for key, val in arg_dict.items():
        print(f"* {key}: {val}")
    print(" ------- ------- ------- ")


def main():

    args = get_args()
    check_params(args)

    input_text = args.input
    output_dir = args.output

    # initialize fastspeech2 model
    text2eml_config = AutoConfig.from_pretrained(args.text2mel_config)
    text2mel_name = args.text2mel_name
    text2mel = TFAutoModel.from_pretrained(
        config=text2eml_config,
        pretrained_path=args.text2mel_path,
        name=text2mel_name
    )

    # initialize melgan model
    vocoder_config = AutoConfig.from_pretrained(args.vocoder_config)
    vocoder_name = args.vocoder_name
    vocoder = TFAutoModel.from_pretrained(
        config=vocoder_config,
        pretrained_path=args.vocoder_path,
        name=vocoder_name
    )

    processor = AutoProcessor.from_pretrained(pretrained_path="./data_prep/kss/kss_mapper.json")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_text, 'r') as f:
        for i, line in enumerate(f):
            input_text = line.strip()
            _, audio = do_synthesis(
                input_text, text2mel, vocoder,
                text2mel_name.upper(), vocoder_name.upper(), processor
            )
            sf.write(f'{output_dir}/line{i}.wav', audio, 22050, "PCM_16")
            print(i, input_text)


if __name__ == '__main__':
    main()
