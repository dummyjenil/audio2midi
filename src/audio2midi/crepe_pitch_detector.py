import warnings
import numpy as np
import torch
import librosa
from torch.nn import functional as F
from tqdm import tqdm
from functools import partial
from huggingface_hub import hf_hub_download
from scipy.stats import triang

###############################################################################
# Constants
###############################################################################

CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan
# Minimum decibel level
MIN_DB = -100.

# Reference decibel level
REF_DB = 20.




###############################################################################
# Probability sequence decoding methods
###############################################################################


def argmax(logits):
    """Sample observations by taking the argmax"""
    bins = logits.argmax(dim=1)

    # Convert to frequency in Hz
    return bins, bins_to_frequency(bins)











###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    return dither(cents)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return quantize_fn(bins).int()


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * torch.log2(frequency / 10.)















###############################################################################
# Pitch thresholding methods
###############################################################################


class At:
    """Simple thresholding at a specified probability value"""

    def __init__(self, value):
        self.value = value

    def __call__(self, pitch, periodicity):
        # Make a copy to prevent in-place modification
        pitch = torch.clone(pitch)

        # Threshold
        pitch[periodicity < self.value] = UNVOICED
        return pitch


class Hysteresis:
    """Hysteresis thresholding"""

    def __init__(self,
                 lower_bound=.19,
                 upper_bound=.31,
                 width=.2,
                 stds=1.7,
                 return_threshold=False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width
        self.stds = stds
        self.return_threshold = return_threshold

    def __call__(self, pitch, periodicity):
        # Save output device
        device = pitch.device

        # Perform hysteresis in log-2 space
        pitch = torch.log2(pitch).detach().flatten().cpu().numpy()

        # Flatten periodicity
        periodicity = periodicity.flatten().cpu().numpy()

        # Ignore confidently unvoiced pitch
        pitch[periodicity < self.lower_bound] = UNVOICED

        # Whiten pitch
        mean, std = np.nanmean(pitch), np.nanstd(pitch)
        pitch = (pitch - mean) / std

        # Require high confidence to make predictions far from the mean
        parabola = self.width * pitch ** 2 - self.width * self.stds ** 2
        threshold = \
            self.lower_bound + np.clip(parabola, 0, 1 - self.lower_bound)
        threshold[np.isnan(threshold)] = self.lower_bound

        # Apply hysteresis to prevent short, unconfident voiced regions
        i = 0
        while i < len(periodicity) - 1:

            # Detect unvoiced to voiced transition
            if periodicity[i] < threshold[i] and \
               periodicity[i + 1] > threshold[i + 1]:

                # Grow region until next unvoiced or end of array
                start, end, keep = i + 1, i + 1, False
                while end < len(periodicity) and \
                      periodicity[end] > threshold[end]:
                    if periodicity[end] > self.upper_bound:
                        keep = True
                    end += 1

                # Force unvoiced if we didn't pass the confidence required by
                # the hysteresis
                if not keep:
                    threshold[start:end] = 1

                i = end

            else:
                i += 1

        # Remove pitch with low periodicity
        pitch[periodicity < threshold] = UNVOICED

        # Unwhiten
        pitch = pitch * std + mean

        # Convert to Hz
        pitch = torch.tensor(2 ** pitch, device=device)[None, :]

        # Optionally return threshold
        if self.return_threshold:
            return pitch, torch.tensor(threshold, device=device)

        return pitch


###############################################################################
# Periodicity thresholding methods
###############################################################################


class Silence:
    """Set periodicity to zero in silent regions"""

    def __init__(self, value=-60):
        self.value = value
        self.a_weighted_weights = self.perceptual_weights()
    def perceptual_weights(self):
        """A-weighted frequency-dependent perceptual loudness weights"""
        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE,n_fft=WINDOW_SIZE)

        # A warning is raised for nearly inaudible frequencies, but it ends up
        # defaulting to -100 db. That default is fine for our purposes.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return librosa.A_weighting(frequencies)[:, None] - REF_DB

    def a_weighted(self,audio, sample_rate, hop_length=None, pad=True):
        """Retrieve the per-frame loudness"""
        # Save device
        device = audio.device

        # Default hop length of 10 ms
        hop_length = sample_rate // 100 if hop_length is None else hop_length

        # Convert to numpy
        audio = audio.detach().cpu().numpy().squeeze(0)

        # Take stft
        stft = librosa.stft(audio,
                            n_fft=WINDOW_SIZE,
                            hop_length=hop_length,
                            win_length=WINDOW_SIZE,
                            center=pad,
                            pad_mode='constant')

        # Compute magnitude on db scale
        db = librosa.amplitude_to_db(np.abs(stft))

        # Apply A-weighting
        weighted = db + self.a_weighted_weights

        # Threshold
        weighted[weighted < MIN_DB] = MIN_DB

        # Average over weighted frequencies
        return torch.from_numpy(weighted.mean(axis=0)).float().to(device)[None]

    def __call__(self,
                 periodicity,
                 audio,
                 sample_rate=SAMPLE_RATE,
                 hop_length=None,
                 pad=True):
        # Don't modify in-place
        periodicity = torch.clone(periodicity)

        # Compute loudness
        loudness = self.a_weighted(
            audio, sample_rate, hop_length, pad)

        # Threshold silence
        periodicity[loudness < self.value] = 0.

        return periodicity







###############################################################################
# Sequence filters
###############################################################################


def mean(signals, win_length=9):
    """Averave filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """

    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)

    # Apply the mask by setting masked elements to zero, or make NaNs zero
    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))

    # Create a ones kernel with the same number of channels as the input tensor
    ones_kernel = torch.ones(signals.size(1), 1, win_length, device=signals.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=win_length // 2,
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=win_length // 2,
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    # Fill zero values with NaNs
    avg_pooled[avg_pooled == 0] = float("nan")

    return avg_pooled.squeeze(1)


def median(signals, win_length):
    """Median filtering for signals containing nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """

    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)

    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))
    padding = win_length // 2

    x = F.pad(masked_x, (padding, padding), mode="reflect")
    mask = F.pad(mask.float(), (padding, padding), mode="constant", value=0)

    x = x.unfold(2, win_length, 1)
    mask = mask.unfold(2, win_length, 1)

    x = x.contiguous().view(x.size()[:3] + (-1,))
    mask = mask.contiguous().view(mask.size()[:3] + (-1,))

    # Combine the mask with the input tensor
    x_masked = torch.where(mask.bool(), x.float(), float("inf")).to(x)

    # Sort the masked tensor along the last dimension
    x_sorted, _ = torch.sort(x_masked, dim=-1)

    # Compute the count of non-masked (valid) values
    valid_count = mask.sum(dim=-1)

    # Calculate the index of the median value for each pooling window
    median_idx = ((valid_count - 1) // 2).clamp(min=0)

    # Gather the median values using the calculated indices
    median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

    # Fill infinite values with NaNs
    median_pooled[torch.isinf(median_pooled)] = float("nan")

    return median_pooled.squeeze(1)


###############################################################################
# Utilities
###############################################################################


def nanfilter(signals, win_length, filter_fn):
    """Filters a sequence, ignoring nan values

    Arguments
        signals (torch.tensor (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window
        filter_fn (function)
            The function to use for filtering

    Returns
        filtered (torch.tensor (shape=(batch, time)))
    """
    # Output buffer
    filtered = torch.empty_like(signals)

    # Loop over frames
    for i in range(signals.size(1)):

        # Get analysis window bounds
        start = max(0, i - win_length // 2)
        end = min(signals.size(1), i + win_length // 2 + 1)

        # Apply filter to window
        filtered[:, i] = filter_fn(signals[:, start:end])

    return filtered


def nanmean(signals):
    """Computes the mean, ignoring nans

    Arguments
        signals (torch.tensor [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (torch.tensor [shape=(batch, time)])
    """
    signals = signals.clone()

    # Find nans
    nans = torch.isnan(signals)

    # Set nans to 0.
    signals[nans] = 0.

    # Compute average
    return signals.sum(dim=1) / (~nans).float().sum(dim=1)


def nanmedian(signals):
    """Computes the median, ignoring nans

    Arguments
        signals (torch.tensor [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (torch.tensor [shape=(batch, time)])
    """
    # Find nans
    nans = torch.isnan(signals)

    # Compute median for each slice
    medians = [nanmedian1d(signal[~nan]) for signal, nan in zip(signals, nans)]

    # Stack results
    return torch.tensor(medians, dtype=signals.dtype, device=signals.device)


def nanmedian1d(signal):
    """Computes the median. If signal is empty, returns torch.nan

    Arguments
        signal (torch.tensor [shape=(time,)])

    Returns
        median (torch.tensor [shape=(1,)])
    """
    return torch.median(signal) if signal.numel() else np.nan


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = triang.rvs(c=0.5,loc=-CENTS_PER_BIN,scale=2 * CENTS_PER_BIN,size=cents.size())
    return cents + cents.new_tensor(noise)

def periodicity(probabilities, bins):
    """Computes the periodicity from the network output and pitch bins"""
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)

    # shape=(batch * time / hop_length, 1)
    bins_stacked = bins.reshape(-1, 1).to(torch.int64)

    # Use maximum logit over pitch bins as periodicity
    periodicity = probs_stacked.gather(1, bins_stacked)

    # shape=(batch, time / hop_length)
    return periodicity.reshape(probabilities.size(0), probabilities.size(2))













class CrepeTorch(torch.nn.Module):

    def __init__(self, model_type='full',model_path=None):
        super().__init__()
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        self.viterbi_transition = transition / transition.sum(axis=1, keepdims=True)
        model_type_importance = {'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32}[model_type]
        out_channels = [n * model_type_importance for n in [32, 4, 4, 4, 8, 16]]
        in_channels = [n * model_type_importance for n in [32, 4, 4, 4, 8]]
        in_channels.insert(0,1)
        self.in_features = 64*model_type_importance
        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = partial(torch.nn.BatchNorm2d,eps=0.0010000000474974513,momentum=0.0)

        # Layer definitions
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = torch.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = torch.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = torch.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = torch.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])

        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=PITCH_BINS)
        if not model_path:
            model_path = hf_hub_download("shethjenil/Audio2Midi_Models",f"crepe_{model_type}.pt")
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def forward(self, x, embed=False):
        # Forward pass through first five layers
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = self.layer(x, self.conv6, self.conv6_BN)

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        # Compute logits
        return torch.sigmoid(self.classifier(x))

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)
        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))

    def viterbi(self,logits):
        """Sample observations using viterbi decoding"""
        # Normalize logits
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)

        # Convert to numpy
        sequences = probs.cpu().numpy()

        # Perform viterbi decoding
        bins = np.array([
            librosa.sequence.viterbi(sequence, self.viterbi_transition).astype(np.int64)
            for sequence in sequences])

        # Convert to pytorch
        bins = torch.tensor(bins, device=probs.device)

        # Convert to frequency in Hz
        return bins, bins_to_frequency(bins)

    def get_device(self):
        return next(self.parameters()).device

    def preprocess(self,audio,sample_rate,hop_length=None,batch_size=None,pad=True):
        """Convert audio to model input

        Arguments
            audio (torch.tensor [shape=(1, time)])
                The audio signals
            sample_rate (int)
                The sampling rate in Hz
            hop_length (int)
                The hop_length in samples
            batch_size (int)
                The number of frames per batch
            pad (bool)
                Whether to zero-pad the audio

        Returns
            frames (torch.tensor [shape=(1 + int(time // hop_length), 1024)])
        """
        # Default hop length of 10 ms
        hop_length = sample_rate // 100 if hop_length is None else hop_length

        # Get total number of frames

        # Maybe pad
        if pad:
            total_frames = 1 + int(audio.size(1) // hop_length)
            audio = torch.nn.functional.pad(
                audio,
                (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        else:
            total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

        # Default to running all frames in a single batch
        batch_size = total_frames if batch_size is None else batch_size

        # Generate batches
        for i in range(0, total_frames, batch_size):

            # Batch indices
            start = max(0, i * hop_length)
            end = min(audio.size(1),
                    (i + batch_size - 1) * hop_length + WINDOW_SIZE)

            # Chunk
            frames = torch.nn.functional.unfold(
                audio[:, None, None, start:end],
                kernel_size=(1, WINDOW_SIZE),
                stride=(1, hop_length))

            # shape=(1 + int(time / hop_length, 1024)
            frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)

            # Place on device
            frames = frames.to(self.get_device())

            # Mean-center
            frames -= frames.mean(dim=1, keepdim=True)

            # Scale
            # Note: during silent frames, this produces very large values. But
            # this seems to be what the network expects.
            frames /= torch.max(torch.tensor(1e-10, device=frames.device),frames.std(dim=1, keepdim=True))

            yield frames

    def postprocess(self,probabilities,fmin=0.,fmax=MAX_FMAX,return_periodicity=False):
        """Convert model output to F0 and periodicity

        Arguments
            probabilities (torch.tensor [shape=(1, 360, time / hop_length)])
                The probabilities for each pitch bin inferred by the network
            fmin (float)
                The minimum allowable frequency in Hz
            fmax (float)
                The maximum allowable frequency in Hz
            viterbi (bool)
                Whether to use viterbi decoding
            return_periodicity (bool)
                Whether to also return the network confidence

        Returns
            pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
            periodicity (torch.tensor [shape=(1, 1 + int(time // hop_length))])
        """
        # Sampling is non-differentiable, so remove from graph
        probabilities = probabilities.detach()

        # Convert frequency range to pitch bin range
        minidx = frequency_to_bins(torch.tensor(fmin))
        maxidx = frequency_to_bins(torch.tensor(fmax),
                                                    torch.ceil)

        # Remove frequencies outside of allowable range
        probabilities[:, :minidx] = -float('inf')
        probabilities[:, maxidx:] = -float('inf')

        # Perform argmax or viterbi sampling
        bins, pitch = self.viterbi(probabilities)

        if not return_periodicity:
            return pitch

        # Compute periodicity from probabilities and decoded pitch bins
        return pitch, periodicity(probabilities, bins)

    def predict(self,audio,sample_rate,hop_length=None,fmin=50.,fmax=MAX_FMAX,return_periodicity=False,batch_size=None,pad=True):
        """Performs pitch estimation

        Arguments
            audio (torch.tensor [shape=(1, time)])
                The audio signal
            sample_rate (int)
                The sampling rate in Hz
            hop_length (int)
                The hop_length in samples
            fmin (float)
                The minimum allowable frequency in Hz
            fmax (float)
                The maximum allowable frequency in Hz
            return_periodicity (bool)
                Whether to also return the network confidence
            batch_size (int)
                The number of frames per batch
            pad (bool)
                Whether to zero-pad the audio

        Returns
            pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
            (Optional) periodicity (torch.tensor
                                    [shape=(1, 1 + int(time // hop_length))])
        """

        results = []
        with torch.no_grad():
            print("prediction started")
            for frames in self.preprocess(audio,sample_rate,hop_length,batch_size,pad):
                # shape=(batch, 360, time / hop_length)
                result = self.postprocess(self.forward(frames, embed=False).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2),fmin,fmax,return_periodicity)
                if isinstance(result, tuple):
                    result = (result[0].to(audio.device),result[1].to(audio.device))
                else:
                    result = result.to(audio.device)
                results.append(result)
            print("prediction finished")
        if return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        return torch.cat(results, 1)

    def predict_from_file(self,audio_file,hop_length=None,fmin=50.,fmax=MAX_FMAX,return_periodicity=False,batch_size=None,pad=True):
        audio, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        return self.predict(torch.from_numpy(audio).unsqueeze(0),sample_rate,hop_length,fmin,fmax,return_periodicity,batch_size,pad)

    def predict_from_file_to_file(self,audio_file,output_pitch_file,output_periodicity_file=None,hop_length=None,fmin=50.,fmax=MAX_FMAX,batch_size=None,pad=True):
        prediction = self.predict_from_file(audio_file,hop_length,fmin,fmax,False,output_periodicity_file is not None,batch_size,pad)
        if output_periodicity_file is not None:
            torch.save(prediction[0].detach(), output_pitch_file)
            torch.save(prediction[1].detach(), output_periodicity_file)
        else:
            torch.save(prediction.detach(), output_pitch_file)

    def predict_from_files_to_files(self,audio_files,output_pitch_files,output_periodicity_files=None,hop_length=None,fmin=50.,fmax=MAX_FMAX,batch_size=None,pad=True):
        if output_periodicity_files is None:
            output_periodicity_files = len(audio_files) * [None]
        for audio_file, output_pitch_file, output_periodicity_file in tqdm(zip(audio_files, output_pitch_files, output_periodicity_files), desc='torchcrepe', dynamic_ncols=True):
            self.predict_from_file_to_file(audio_file,output_pitch_file,None,output_periodicity_file,hop_length,fmin,fmax,batch_size,pad)

    def embedding(self,audio,sample_rate,hop_length=None,batch_size=None,pad=True):
        """Embeds audio to the output of CREPE's fifth maxpool layer

        Arguments
            audio (torch.tensor [shape=(1, time)])
                The audio signals
            sample_rate (int)
                The sampling rate in Hz
            hop_length (int)
                The hop_length in samples
            batch_size (int)
                The number of frames per batch
            pad (bool)
                Whether to zero-pad the audio

        Returns
            embedding (torch.tensor [shape=(1,
                                            1 + int(time // hop_length), 32, -1)])
        """
        # shape=(batch, time / hop_length, 32, embedding_size)
        with torch.no_grad():
            return torch.cat([self.forward(frames, embed=True).reshape(audio.size(0), frames.size(0), 32, -1).to(audio.device) for frames in self.preprocess(audio,sample_rate,hop_length,batch_size,pad)], 1)

    def embedding_from_file(self,audio_file,hop_length=None,batch_size=None,pad=True):
        audio, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        return self.embed(torch.from_numpy(audio).unsqueeze(0),sample_rate,hop_length,batch_size,pad)

    def embedding_from_file_to_file(self,audio_file,output_file,hop_length=None,batch_size=None,pad=True):
        with torch.no_grad():
            torch.save(self.embed_from_file(audio_file,hop_length,batch_size,pad).detach(), output_file)

    def embedding_from_files_to_files(self,audio_files,output_files,hop_length=None,batch_size=None,pad=True):
        for audio_file, output_file in tqdm(zip(audio_files, output_files), desc='torchcrepe', dynamic_ncols=True):
            self.embed_from_file_to_file(audio_file,output_file,hop_length,batch_size,pad)












from hmmlearn.hmm import CategoricalHMM
from typing import Callable
from numpy.lib.stride_tricks import as_strided
from pretty_midi_fix import PrettyMIDI , PitchBend , Note ,Instrument
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, TensorDataset

class Crepe():

    def __init__(self,model_type="full",model_path=None):
        self.model = CrepeTorch(model_type,model_path)
        self.cents_mapping=(np.linspace(0, 7180, 360) + 1997.3794084376191)

    def to_local_average_cents(self, salience, center=None):
        if isinstance(salience, torch.Tensor):
            salience = salience.numpy()

        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(salience * self.cents_mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum
        if salience.ndim == 2:
            return np.array([self.to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")

    def to_viterbi_cents(self,salience):
        if isinstance(salience, torch.Tensor):
            salience = salience.numpy()
        starting = np.ones(360) / 360
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / np.sum(transition, axis=1)[:, None]
        self_emission = 0.1
        emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) * ((1 - self_emission) / 360))
        model = CategoricalHMM(360, starting, transition)
        model.startprob_, model.transmat_, model.emissionprob_ = starting, transition, emission
        observations = np.argmax(salience, axis=1)
        path = model.predict(observations.reshape(-1, 1), [len(observations)])
        return np.array([self.to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])

    def get_activation(self,audio:np.ndarray,center, step_size, progress_callback,batch_size):
        if center:
            audio = np.pad(audio, 512, mode='constant', constant_values=0)
        hop_length = int(16000 * step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        frames = as_strided(audio, shape=(1024, n_frames),strides=(audio.itemsize, hop_length * audio.itemsize))
        frames = frames.transpose().copy()
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)
        device = self.model.get_device()
        all_outputs = []
        all_batch = list(DataLoader(TensorDataset(torch.from_numpy(frames)), batch_size=batch_size, shuffle=False))
        total_batch = len(all_batch)
        with torch.no_grad():
            for i , batch in enumerate(all_batch):
                inputs = batch[0].to(device)
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())
                if progress_callback:
                    progress_callback(i,total_batch)
        return torch.cat(all_outputs, dim=0)

    def model_predict(self,audio:np.ndarray,viterbi, center, step_size,progress_callback,batch_size):
        activation = self.get_activation(audio.astype(np.float32), center, step_size,progress_callback,batch_size)
        confidence = activation.max(axis=1).values # Access the values from the named tuple
        cents = self.to_viterbi_cents(activation) if viterbi else self.to_local_average_cents(activation)
        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0
        time = np.arange(confidence.shape[0]) * step_size / 1000.0
        return time, frequency, confidence

    def predict(self,audio_path,viterbi=False, center=True, step_size=10,min_confidence=0.8,batch_size=32,progress_callback: Callable[[int, int], None] = None,output_file= "output.mid"):
        time, frequency, confidence = self.model_predict(librosa.load(audio_path, sr=16000, mono=True)[0],viterbi,center,step_size,progress_callback,batch_size)
        mask = confidence > min_confidence
        times = time[mask]
        frequencies = frequency[mask]
        midi_floats = 69 + 12 * np.log2(frequencies / 440.0)
        midi_notes = np.round(midi_floats).astype(int)
        pitch_offsets = midi_floats - midi_notes  # in semitones
        midi = PrettyMIDI()
        instrument = Instrument(program=40)  # e.g., Violin for pitch bend demo
        if len(times) > 0:
            current_note = midi_notes[0]
            note_start = times[0]
            for i in range(1, len(times)):
                if midi_notes[i] != current_note or i == len(times) - 1:
                    note_end = times[i]
                    if 0 <= current_note <= 127:
                        note = Note(velocity=100,pitch=int(current_note),start=note_start,end=note_end)
                        instrument.notes.append(note)
                        seg_mask = (times >= note_start) & (times <= note_end)
                        seg_times = times[seg_mask]
                        seg_offsets = pitch_offsets[seg_mask]
                        for t, offset in zip(seg_times, seg_offsets):
                            # Assuming pitch bend range is +/- 2 semitones
                            bend_value = int(offset / 2.0 * 8192)  # Scale to -8192 to +8191
                            bend_value = np.clip(bend_value, -8192, 8191)
                            pb = PitchBend(pitch=bend_value, time=t)
                            instrument.pitch_bends.append(pb)
                    current_note = midi_notes[i]
                    note_start = times[i]
        midi.instruments.append(instrument)
        midi.write(output_file)
        return output_file
