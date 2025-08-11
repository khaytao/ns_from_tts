import os
import numpy as np
from scipy.io import wavfile


def _calculate_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Generate white Gaussian noise for a given signal to achieve the desired SNR (in dB).

    Args:
        signal: Audio signal as a float32/float64 numpy array in the range [-1, 1].
        snr_db: Desired signal-to-noise ratio in decibels.

    Returns:
        White Gaussian noise array with the same shape as ``signal``.
    """
    # Compute RMS (root mean square) power of the clean signal
    rms_signal = np.sqrt(np.mean(signal ** 2)) + 1e-12  # avoid division by zero

    # Desired RMS of the noise to obtain the given SNR
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, size=signal.shape).astype(signal.dtype)
    return noise


def _normalize_int(signal: np.ndarray, dtype) -> np.ndarray:
    """Convert floating point signal in [-1, 1] to an integer dtype."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = np.clip(signal, -1.0, 1.0)
        scaled = (scaled * info.max).astype(dtype)
        return scaled
    return signal.astype(dtype)


def add_noise(input_dir: str, output_dir: str, snr: float) -> None:
    """Add white Gaussian noise to all ``.wav`` files in *input_dir* at the given *snr* (dB).

    The noisy files are written to *output_dir* using the same filenames.

    Args:
        input_dir: Directory containing clean ``.wav`` files.
        output_dir: Directory where noisy wav files will be stored (created if absent).
        snr: Desired signal-to-noise ratio in decibels. Higher values mean less noise.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in '{input_dir}'. Nothing to do.")
        return

    for filename in wav_files:
        in_path = os.path.join(input_dir, filename)
        sr, data = wavfile.read(in_path)

        # Handle stereo or mono; convert to float in [-1, 1]
        dtype = data.dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data_float = data.astype(np.float32) / info.max
        else:
            data_float = data.astype(np.float32)

        noise = _calculate_noise(data_float, snr)
        noisy_float = data_float + noise

        # Ensure we don't exceed [-1, 1] after noise addition
        noisy_float = np.clip(noisy_float, -1.0, 1.0)

        # Convert back to original dtype for saving
        noisy_data = _normalize_int(noisy_float, dtype)

        out_path = os.path.join(output_dir, filename)
        wavfile.write(out_path, sr, noisy_data)
        print(f"Saved noisy file to {out_path}")


if __name__ == "__main__":

    example_input = r"D:\projects\NSTTS\datasets\LJSpeech-1.1_mini\wavs"
    example_output = r"D:\projects\NSTTS\datasets\LJSpeech-1.1_mini\noisy_wavs"
    example_snr = 10  # dB
    add_noise(example_input, example_output, example_snr)