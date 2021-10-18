FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update && apt-get install -y \
        zsh tmux wget git libsndfile1 sox \
    && \
    pip install \
        ipython==7.16.1  \
        h5py==2.10.0 \
        git+https://github.com/TensorSpeech/TensorflowTTS.git@4b9a446152f936505ce3baee0f46d8f75c19c32f \
        git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate
        
RUN mkdir /workspace
WORKDIR /workspace
