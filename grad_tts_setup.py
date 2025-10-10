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
audio_dir = get_audio_dir()
import controlnet_params
file_dirs = [controlnet_params.train_filelist_path, controlnet_params.valid_filelist_path, controlnet_params.test_filelist_path]
for d in file_dirs:
    generate_filelist(d, audio_dir)

# python inference.py -f ./resources/filelists/synthesis.txt -c ./checkpts/grad-tts.pt

# for training, edit params.py then run python train.py

