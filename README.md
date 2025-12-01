# ns_from_tts

This is a project to turn a text to speech (TTS) model into a noise suppression (NS) model,
using ControlNet architecture on top of the Grad-TTS model.

This project was forked from https://github.com/huawei-noah/Speech-Backbones/tree/main
commit 7782c7a

## Project Setup

Follow these setup instructions from Grad-TTS

1. Download hifi-gan and grad-tts weights into the checkpts directory

2. run !cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..

Download the LJSpeech dataset

run grad_tts_setup.py

you can set os.environ['AUDIO_DIR'] = files_path before. If not, files_path would be read from input()  

files_path is a file path to your LJSpeech dataset





- `data.py`
- 
- `mel_comparison.py`


## Training

Edit - `controlnet_params.py` and then run - `train_with_controlnet.py`

## Inference
In order to generate audio examples, run: `controlnet_inference_with_comparisons.py`
mandatory arguments:

    parser.add_argument('-f', '--file', type=str, required=True, help='Path to filelist (wav|text) to analyse')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to GradTTS_NS checkpoint (with ControlNet)')

For each analysed file, this script will generate:
1. A copy of the original file
2. A version generated strictly from the Text-To-Speech model
3. A version generated from the Controlnet

### Analysis scripts
- `run_comparisons.py`
Will calculate DTW for each test file, for both the TTS and the Control model. It will save the output as a Json

- `analyze_comparisons.py`
Will calculate statistics for the above Json output


-`get_baseline.py`
Will calculate the DTW between a random test sample and N random noise signals

## Notable code changes from Grad-TTS

### Model
- `src/model/diffusion_with_controlnet.py`
Here is the controlnet implementation. We created a class that is inherited from the original diffusion class, and added the controlnet specific layers and methods.

- `src/model/tts_ns.py`
Here is the controlled model. Note changes to the forward and compute_loss function.

### Dataloader
We added TextMelNoisyMelDataset that instead of loading a text_encoding, mel pair loads a text_encoding, clean_mel, noisy_mel triplet. It has an SNR_DB input parameter, that controls the SNR of the noisy_mel. 