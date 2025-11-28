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
    """Process a single batch through both models"""
    x = batch['x'].to(device)
    x_lengths = torch.tensor([x.size(1)], device=device)
    y = batch['y'].to(device)
    c = batch['c'].to(device)
    c_lengths = torch.tensor([c.size(2)], device=device)

    with torch.no_grad():
        # Base model
        _, y_dec_base, _ = base_model(
            x.unsqueeze(0), x_lengths, c.unsqueeze(0), c_lengths,
            n_timesteps=timesteps, temperature=1.0, stoc=False,
            use_mas=True, clean=y.unsqueeze(0)
        )

        # Control model
        _, y_dec_control, _ = control_model(
            x.unsqueeze(0), x_lengths, c.unsqueeze(0), c_lengths,
            n_timesteps=timesteps, temperature=1.0, stoc=False,
            use_mas=True, clean=y.unsqueeze(0)
        )

        # Compare
        result = compare_to_reference(
            y.squeeze(0),
            y_dec_control.squeeze(0),
            y_dec_base.squeeze(0),
            name1="control",
            name2="base"
        )

    return result

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
    base_model = load_model(args.grad_checkpoint, device)
    control_model = load_model(args.checkpoint, device)

    # Load dataset
    print("Loading dataset...")
    dataset = TextMelNoisyMelDataset(
        filelist_path=args.file,
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
    with torch.no_grad():
        with tqdm(test_loader, total=len(dataset) // batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                result = process_batch(batch, base_model, control_model, device, args.timesteps)
                all_results.append(result)

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