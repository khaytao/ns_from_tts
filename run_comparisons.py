import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchaudio as ta
from tqdm import tqdm

# Add hifi-gan to path
sys.path.append('./hifi-gan')
from env import AttrDict
from models import Generator as HiFiGAN

# Import local modules
from model.tts_ns import GradTTS_NS
from data import TextMelNoisyMelDataset, TextMelNoisyMelBatchCollate
from mel_comparison import compare_to_reference
from text.symbols import symbols
import controlnet_params as params

# Constants
BASE_PRETRAINED_PATH = './checkpts/controlnet_model_base_grad.pt'
HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

# Audio params from controlnet_params
n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max
batch_size = params.batch_size

spk = None


def load_model(checkpoint_path, device='cuda'):
    """Load GradTTS_NS model from checkpoint"""
    model = GradTTS_NS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                             params.n_enc_channels, params.filter_channels,
                             params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                             params.enc_kernel, params.enc_dropout, params.window_size,
                             params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    if hasattr(model.decoder, 'estimator'):
        model.decoder.estimator.is_initialized = True
    return model.eval().to(device)

def process_batch(batch, base_model, control_model, device, timesteps=10):
    # Unpack the collated batch just like in train_with_controlnet / inference
    # Adjust the number/order of items if your collate returns speakers etc.
    x, x_lengths, y, y_lengths, c, c_lengths = batch

    # Move to device
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    y = y.to(device)
    y_lengths = y_lengths.to(device)
    c = c.to(device)
    c_lengths = c_lengths.to(device)

    with torch.no_grad():
        # Base model on the *batched* inputs
        _, y_dec_base, _ = base_model(
            x, x_lengths, c, c_lengths,
            n_timesteps=timesteps, temperature=1.0, stoc=False,
            use_mas=True, clean=y
        )

        # Control model on the same batch
        _, y_dec_control, _ = control_model(
            x, x_lengths, c, c_lengths,
            n_timesteps=timesteps, temperature=1.0, stoc=False,
            use_mas=True, clean=y
        )

        # Now compare on the *full batch*
        # compare_to_reference should either:
        #   - handle batched tensors, or
        #   - you loop over batch dimension inside here
        # Here’s a simple per-sample loop:
        batch_size = y.shape[0]
        results = []
        for i in range(batch_size):
            r = compare_to_reference(
                y[i],
                y_dec_control[i],
                y_dec_base[i],
                name1="control",
                name2="base"
            )
            results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare clean reference audio with baseline TTS synthesis (no ControlNet).')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to filelist (wav|text) to analyse')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to GradTTS_NS checkpoint (with ControlNet)')
    parser.add_argument('-gradc', '--grad_checkpoint', type=str, default=BASE_PRETRAINED_PATH,
                        help='Path to pretrained GradTTS_NS checkpoint for baseline (default: %(default)s)')
    parser.add_argument('-n', '--num_samples', type=int, default=10, help='Number of random samples to analyse')
    parser.add_argument('-t', '--timesteps', type=int, default=10, help='Number of diffusion reverse steps')
    parser.add_argument('-s', '--speaker_id', type=int, default=None, help='Speaker id for multi-speaker model')
    parser.add_argument('-o', '--outdir', type=str, default='./analysis', help='Directory to write results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Setup output
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    base_generator = load_model(args.grad_checkpoint, device)
    control_generator = load_model(args.checkpoint, device)

    # Load dataset
    print("Loading dataset...")
    dataset = TextMelNoisyMelDataset(
        filelist_path=params.test_filelist_path,
        cmudict_path=params.cmudict_path,
        add_blank=params.add_blank,
        n_fft=params.n_fft,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=params.win_length,
        f_min=params.f_min,
        f_max=params.f_max,
        snr_db=params.snr_db
    )
    batch_collate = TextMelNoisyMelBatchCollate()
    test_loader = DataLoader(dataset=dataset, batch_size=params.test_size,
                             collate_fn=batch_collate, drop_last=True,
                             num_workers=4, shuffle=False)
    # Process all samples
    print("Processing samples...")
    all_results = []
    # with torch.no_grad():
    #     # Much simpler and safer: total=len(test_loader)
    #     with tqdm(test_loader, total=len(test_loader)) as progress_bar:
    #         for batch_idx, batch in enumerate(progress_bar):
    #             batch_results = process_batch(batch, base_model, control_model, device, args.timesteps)
    #             all_results.extend(batch_results)

    with torch.no_grad():
        with tqdm(test_loader, total=len(dataset) // batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):

                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                c, c_lengths = batch['c'].cuda(), batch['c_lengths'].cuda()

                y_enc_base, y_dec_base, _ = base_generator.forward(x, x_lengths,
                                                                   c, c_lengths,
                                                                   n_timesteps=args.timesteps,
                                                                   temperature=1.5, stoc=False,
                                                                   spk=spk, length_scale=1.0,
                                                                   use_mas=True, clean=y)

                y_enc_ctrl, y_dec_ctrl, _ = control_generator.forward(x, x_lengths,
                                                                   c, c_lengths,
                                                                   n_timesteps=args.timesteps,
                                                                   temperature=1.5, stoc=False,
                                                                   spk=spk, length_scale=1.0,
                                                                   use_mas=True, clean=y)

                comparison_result = compare_to_reference(y, torch.squeeze(y_dec_ctrl),
                                                         torch.squeeze(y_dec_base), name1="with control", name2="base")

                all_results.append(comparison_result)
    # Compute statistics after processing
    if not all_results:
        print("No samples processed, not writing stats.")
        stats = {}
    else:
        metrics = defaultdict(list)
        for r in all_results:
            for k, v in r.items():
                metrics[k].append(v)

        print("\n=== Results ===")
        stats = {}
        for metric, values in metrics.items():
            values = np.array(values)
            stats[metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'control_better': float((values > 0).mean())
            }
            print(f"\n{metric}:")
            print(f"  Mean ± Std: {stats[metric]['mean']:.4f} ± {stats[metric]['std']:.4f}")
            print(f"  Range: [{stats[metric]['min']:.4f}, {stats[metric]['max']:.4f}]")
            print(f"  Control better: {stats[metric]['control_better']:.1%} of samples")

    # Save results
    results_file = out_dir / 'comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()