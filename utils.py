import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn import functional as F


def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert (
                dir_path.exists()
            ), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)



class ExecutionTime:
    """
    Count execution time.

    Examples:
        timer = ExecutionTime()
        ...
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num
    


def stft(y, n_fft, hop_length, win_length):
    """Wrapper of the official torch.stft for single-channel and multi-channel.

    Args:
        y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
        n_fft: number of FFT
        hop_length: hop length
        win_length: hanning window size

    Shapes:
        mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]

    Returns:
        mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
    """
    num_dims = y.dim()
    assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

    batch_size = y.shape[0]
    num_samples = y.shape[-1]

    if num_dims == 3:
        y = y.reshape(-1, num_samples)  # [B * C ,T]

    complex_stft = torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=y.device),
        return_complex=True,
    )
    _, num_freqs, num_frames = complex_stft.shape

    if num_dims == 3:
        complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

    mag = torch.abs(complex_stft)
    phase = torch.angle(complex_stft)
    real = complex_stft.real
    imag = complex_stft.imag
    return mag, phase, real, imag


def istft(features, n_fft, hop_length, win_length, length=None, input_type="complex"):
    """Wrapper of the official torch.istft.

    Args:
        features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
        length: expected length of istft
        use_mag_phase: use mag and phase as the input ("features")

    Returns:
        single-channel speech of shape [B, T]
    """
    if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
        assert isinstance(features, tuple) or isinstance(features, list)
        real, imag = features
        features = torch.complex(real, imag)
    elif input_type == "complex":
        # assert isinstance(features, torch.ComplexType)
        assert isinstance(features, complex)
    elif input_type == "mag_phase":
        # the feature is (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        raise NotImplementedError(
            "Only 'real_imag', 'complex', and 'mag_phase' are supported"
        )

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=features.device),
        length=length,
    )


class Istft(nn.Module):
    # def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
    def __init__(self):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = 320
        self.hop_inv = 160
        self.window_inv = torch.hann_window(320)
        self.w_inv: torch.Tensor
        self.win_len = 320
        # assert window_inv.shape[0] == n_fft_inv
        # self.register_buffer("w_inv", self.window_inv)

    def forward(self, input: Tensor, device):
        # Input shape: [B, * T, F, (2)]
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        # Even though this is not the DF implementation, it numerical sufficiently close.
        # Pad one extra step at the end to get original signal length
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.window_inv.to(device),
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


def as_complex(x: Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def mag_phase(complex_tensor):
    mag, phase = torch.abs(complex_tensor), torch.angle(complex_tensor)
    return mag, phase


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def load_wav(file, sr=16000):
    if len(file) == 2:
        return file[-1]
    else:
        return librosa.load(os.path.abspath(file), mono=False, sr=sr)[0]


def aligned_subsample(data_a, data_b, sub_sample_length):
    """
    Start from a random position and take a fixed-length segment from two speech samples

    Notes
        Only support one-dimensional speech signal (T,) and two-dimensional spectrogram signal (F, T)

        Only support subsample in the last axis.
    """
    assert data_a.shape[-1] == data_b.shape[-1], "Inconsistent dataset size."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        # data_a = data_a[..., start: end]
        return data_a[..., start:end], data_b[..., start:end]
    elif data_a.shape[-1] < sub_sample_length:
        length = data_a.shape[-1]
        pad_size = sub_sample_length - length
        pad_width = [(0, 0)] * (data_a.ndim - 1) + [(0, pad_size)]
        data_a = np.pad(data_a, pad_width=pad_width, mode="constant", constant_values=0)
        data_b = np.pad(data_b, pad_width=pad_width, mode="constant", constant_values=0)
        return data_a, data_b
    else:
        return data_a, data_b


def power_compression(power_stft, trans=False): 
    """
    input: [B, F , T] tensor

    returns: [B , F , T] 
    """
    if trans:  
        power_stft = power_stft.transpose(1,2)
    
    power_stft = torch.pow(power_stft, 0.8)

    return power_stft

def inverse_power_compression(power_stft, trans=False): 
    """
    input: [B, F , T]  tensor

    returns: [B ,F , T] 
    """
    if trans:  
        power_stft = power_stft.transpose(1,2)
    
    power_stft = torch.pow(power_stft, 1.25)

    return power_stft    
    


