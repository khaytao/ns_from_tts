# ensure torch is installed properly
# pip install -r requirements.txt
"""
Here are packages I added to  requirements.txt:
tensorboard
torchaudio
"""

# create ./model/monotonic_align/model/monotonic_align
import os

if not os.path.exists('./model/'):
    raise FileNotFoundError("Directory ./model does not exist, cannot setup model/monotonic_align")
if not os.path.exists('./model/monotonic_align'):
    raise FileNotFoundError("Directory ./model/monotonic_align does not exist, cannot setup model/monotonic_align")

if not os.path.exists('./model/monotonic_align/model'):
    os.mkdir('./model/monotonic_align/model')

if not os.path.exists('./model/monotonic_align/model/monotonic_align'):
    os.mkdir('./model/monotonic_align/model/monotonic_align')

# create appropriate dataset

# python inference.py -f ./resources/filelists/synthesis.txt -c ./checkpts/grad-tts.pt

# for training, edit params.py then run python train.py

