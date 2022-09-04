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

from glob import glob


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        default="models/TF/FASTSPEECH2/",
        help='input keras dir'
    )

    # parser.add_argument(
    #     '--name',
    #     default="models/fastspeech2/v1/model-200000.h5",
    #     choices=['hifi_gan', 'fastspeech2', 'multib'],
    #     help='model dir'
    # )

    parser.add_argument(
        '--output',
        default="output",
        help='output tensorflow saved dir'
    )

    args = parser.parse_args()

    return args


def check_params(arg):
    arg_dict = vars(arg)
    print(" ------- PARAMS ------- ")
    for key, val in arg_dict.items():
        print(f"* {key}: {val}")
    print(" ----------------------- ")


def main():

    args = get_args()
    check_params(args)

    input_dir = args.input
    output_dir = args.output

    model_list = sorted(glob(f"{input_dir}/*.h5"))
    assert len(model_list) != 0, f"There is no model file in {input_dir}"
    model_path = model_list[-1]

    config_list = [file for file in glob(f"{input_dir}/*") if "yaml" in file or 'yml' in file]
    assert len(config_list) != 0, f"There is no config file in {input_dir}"
    config_path = config_list[-1]

    # initialize model
    config = AutoConfig.from_pretrained(config_path)

    model = TFAutoModel.from_pretrained(
        config=config,
        pretrained_path=model_path,
    )
    model.save(output_dir)
    print(f"PARAMs CHECK\n* model_path: {model_path}\n* config_path: {config_path}")
    print(f"DONE !")


if __name__ == '__main__':
    main()
