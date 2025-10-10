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

# if not os.path.exists('./model/'):
#     raise FileNotFoundError("Directory ./model does not exist, cannot setup model/monotonic_align")
# if not os.path.exists('./model/monotonic_align'):
#     raise FileNotFoundError("Directory ./model/monotonic_align does not exist, cannot setup model/monotonic_align")
#
# if not os.path.exists('./model/monotonic_align/model'):
#     os.mkdir('./model/monotonic_align/model')
#
# if not os.path.exists('./model/monotonic_align/model/monotonic_align'):
#     os.mkdir('./model/monotonic_align/model/monotonic_align')

# create appropriate dataset
from pathlib import Path
audio_dir = get_audio_dir()
from controlnet_params import train_filelist_path, valid_filelist_path, test_filelist_path
file_dirs = [train_filelist_path, valid_filelist_path, test_filelist_path]
for d in file_dirs:
    # print(f"🔧 {d} -> {d}_fixed.txt", Path(d).resolve())
    generate_filelist(Path(d).resolve(), audio_dir)

# python inference.py -f ./resources/filelists/synthesis.txt -c ./checkpts/grad-tts.pt

# for training, edit params.py then run python train.py

