# ensure torch is installed properly
# pip install -r requirements.txt
"""
Here are packages I added to  requirements.txt:
tensorboard
torchaudio
"""

# create ./model/monotonic_align/model/monotonic_align
import os
from fix_path_to_dataset import generate_filelist, get_audio_dir


# create appropriate dataset
from pathlib import Path
from controlnet_params import filelist_dir
audio_dir = get_audio_dir()

generate_filelist(Path(filelist_dir).resolve(), audio_dir)

# python inference.py -f ./resources/filelists/synthesis.txt -c ./checkpts/grad-tts.pt

# for training, edit params.py then run python train.py

