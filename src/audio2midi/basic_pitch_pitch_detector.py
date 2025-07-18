from collections import defaultdict
from typing import DefaultDict , List, Optional , Dict , Tuple , Callable
from librosa import load as librosa_load,midi_to_hz,hz_to_midi
from librosa.core import frames_to_time
from librosa.util import frame as librosa_util_frame
from pretty_midi_fix import PrettyMIDI , PitchBend , Instrument,Note
from scipy.signal.windows import gaussian
from scipy.signal import argrelmax
from huggingface_hub import hf_hub_download
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torch
import math

from nnAudio.features import CQT2010v2

FFT_HOP = 256
AUDIO_SAMPLE_RATE = 22050
AUDIO_WINDOW_LENGTH = 2  
NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3
ANNOTATIONS_BASE_FREQUENCY = 27.5  
ANNOTATIONS_N_SEMITONES = 88  
AUDIO_WINDOW_LENGTH = 2
MIDI_OFFSET = 21
N_PITCH_BEND_TICKS = 8192
MAX_FREQ_IDX = 87
N_OVERLAPPING_FRAMES = 30
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
N_FFT = 8 * FFT_HOP
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN
MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))

def frame_with_pad(x: np.array, frame_length: int, hop_size: int) -> np.array:
    """
    Extends librosa.util.frame with end padding if required, similar to 
    tf.signal.frame(pad_end=True).

    Returns:
        framed_audio: tensor with shape (n_windows, AUDIO_N_SAMPLES)
    """
    n_frames = int(np.ceil((x.shape[0] - frame_length) / hop_size)) + 1
    n_pads = (n_frames - 1) * hop_size + frame_length - x.shape[0]
    x = np.pad(x, (0, n_pads), mode="constant")
    framed_audio = librosa_util_frame(x, frame_length=frame_length, hop_length=hop_size)
    return framed_audio


def window_audio_file(audio_original: np.array, hop_size: int) -> Tuple[np.array, List[Dict[str, int]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    audio_windowed = frame_with_pad(audio_original, AUDIO_N_SAMPLES, hop_size)
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times


def get_audio_input(
    audio_path: str, overlap_len: int, hop_size: int
) -> Tuple[Tensor, List[Dict[str, int]], int]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)

    audio_original, _ = librosa_load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, original_length


def unwrap_output(output: Tensor, audio_original_length: int, n_overlapping_frames: int) -> np.array:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    raw_output = output.cpu().detach().numpy()
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]

    output_shape = raw_output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length




def model_output_to_notes(
    output: Dict[str, np.array],
    onset_thresh: float,
    frame_thresh: float,
    infer_onsets: bool = True,
    min_note_len: int = 11,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    include_pitch_bends: bool = True,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = 120,
) -> PrettyMIDI:
    """Convert model output to MIDI

    Args:
        output: A dictionary with shape
            {
                'frame': array of shape (n_times, n_freqs),
                'onset': array of shape (n_times, n_freqs),
                'contour': array of shape (n_times, 3*n_freqs)
            }
            representing the output of the basic pitch model.
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        min_note_len: The minimum allowed note length in frames.
        min_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        max_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        include_pitch_bends: If True, include pitch bends.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.

    Returns:
        midi : PrettyMIDI object
        note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
    """
    frames = output["note"]
    onsets = output["onset"]
    contours = output["contour"]

    estimated_notes = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        infer_onsets=infer_onsets,
        min_note_len=min_note_len,
        min_freq=min_freq,
        max_freq=max_freq,
        melodia_trick=melodia_trick,
    )
    if include_pitch_bends:
        estimated_notes_with_pitch_bend = get_pitch_bends(contours, estimated_notes)
    else:
        estimated_notes_with_pitch_bend = [(note[0], note[1], note[2], note[3], None) for note in estimated_notes]

    times_s = model_frames_to_time(contours.shape[0])
    estimated_notes_time_seconds = [
        (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
    ]

    return note_events_to_midi(estimated_notes_time_seconds, multiple_pitch_bends, midi_tempo)


def midi_pitch_to_contour_bin(pitch_midi: int) -> np.array:
    """Convert midi pitch to conrresponding index in contour matrix

    Args:
        pitch_midi: pitch in midi

    Returns:
        index in contour matrix

    """
    pitch_hz = midi_to_hz(pitch_midi)
    return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY)


def get_pitch_bends(
    contours: np.ndarray, note_events: List[Tuple[int, int, int, float]], n_bins_tolerance: int = 25
) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
    """Given note events and contours, estimate pitch bends per note.
    Pitch bends are represented as a sequence of evenly spaced midi pitch bend control units.
    The time stamps of each pitch bend can be inferred by computing an evenly spaced grid between
    the start and end times of each note event.

    Args:
        contours: Matrix of estimated pitch contours
        note_events: note event tuple
        n_bins_tolerance: Pitch bend estimation range. Defaults to 25.

    Returns:
        note events with pitch bends
    """
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = gaussian(window_length, std=5)
    note_events_with_pitch_bends = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        freq_start_idx = np.max([freq_idx - n_bins_tolerance, 0])
        freq_end_idx = np.min([N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1])

        pitch_bend_submatrix = (
            contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            * freq_gaussian[
                np.max([0, n_bins_tolerance - freq_idx]) : window_length
                - np.max([0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)])
            ]
        )
        pb_shift = n_bins_tolerance - np.max([0, n_bins_tolerance - freq_idx])

        bends: Optional[List[int]] = list(
            np.argmax(pitch_bend_submatrix, axis=1) - pb_shift
        )  # this is in units of 1/3 semitones
        note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return note_events_with_pitch_bends


def note_events_to_midi(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
    multiple_pitch_bends: bool = False,
    midi_tempo: float = 120,
) -> PrettyMIDI:
    """Create a pretty_midi_fix object from note events

    Args:
        note_events : list of tuples [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
            where amplitude is a number between 0 and 1
        multiple_pitch_bends : If True, allow overlapping notes to have pitch bends
            Note: this will assign each pitch to its own midi instrument, as midi does not yet
            support per-note pitch bends

    Returns:
        PrettyMIDI() object

    """
    mid = PrettyMIDI(initial_tempo=midi_tempo)
    if not multiple_pitch_bends:
        note_events_with_pitch_bends = drop_overlapping_pitch_bends(note_events_with_pitch_bends)
    instruments: DefaultDict[int, Instrument] = defaultdict(
        lambda: Instrument(program=40)
    )
    for start_time, end_time, note_number, amplitude, pitch_bend in note_events_with_pitch_bends:
        instrument = instruments[note_number] if multiple_pitch_bends else instruments[0]
        note = Note(
            velocity=int(np.round(127 * amplitude)),
            pitch=note_number,
            start=start_time,
            end=end_time,
        )
        instrument.notes.append(note)
        if not pitch_bend:
            continue
        pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))
        pitch_bend_midi_ticks = np.round(np.array(pitch_bend) * 4096 / CONTOURS_BINS_PER_SEMITONE).astype(int)
        # This supports pitch bends up to 2 semitones
        # If we estimate pitch bends above/below 2 semitones, crop them here when adding them to the midi file
        pitch_bend_midi_ticks[pitch_bend_midi_ticks > N_PITCH_BEND_TICKS - 1] = N_PITCH_BEND_TICKS - 1
        pitch_bend_midi_ticks[pitch_bend_midi_ticks < -N_PITCH_BEND_TICKS] = -N_PITCH_BEND_TICKS
        for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend_midi_ticks):
            instrument.pitch_bends.append(PitchBend(pb_midi, pb_time))
    mid.instruments.extend(instruments.values())

    return mid


def drop_overlapping_pitch_bends(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]]
) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
    """Drop pitch bends from any notes that overlap in time with another note"""
    note_events = sorted(note_events_with_pitch_bends)
    for i in range(len(note_events) - 1):
        for j in range(i + 1, len(note_events)):
            if note_events[j][0] >= note_events[i][1]:  # start j > end i
                break
            note_events[i] = note_events[i][:-1] + (None,)  # last field is pitch bend
            note_events[j] = note_events[j][:-1] + (None,)

    return note_events


def get_infered_onsets(onsets: np.array, frames: np.array, n_diff: int = 2) -> np.array:
    """Infer onsets from large changes in frame amplitudes.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)  # rescale to have the same max as onsets

    max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences

    return max_onsets_diff


def constrain_frequency(
    onsets: np.array, frames: np.array, max_freq: Optional[float], min_freq: Optional[float]
) -> Tuple[np.array, np.array]:
    """Zero out activations above or below the max/min frequencies

    Args:
        onsets: Onset activation matrix (n_times, n_freqs)
        frames: Frame activation matrix (n_times, n_freqs)
        max_freq: The maximum frequency to keep.
        min_freq: the minimum frequency to keep.

    Returns:
       The onset and frame activation matrices, with frequencies outside the min and max
       frequency set to 0.
    """
    if max_freq is not None:
        max_freq_idx = int(np.round(hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:] = 0
        frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx] = 0
        frames[:, :min_freq_idx] = 0

    return onsets, frames


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = frames_to_time(
        np.arange(n_frames),
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
    )
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018  # this is a magic number, but it's needed for this to align properly
    times = original_times - (window_offset * window_numbers)
    return times


def output_to_notes_polyphonic(
    frames: np.array,
    onsets: np.array,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: Optional[float],
    min_freq: Optional[float],
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> List[Tuple[int, int, int, float]]:
    """Decode raw model output to polyphonic note events

    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        max_freq: Maximum allowed output frequency, in Hz.
        min_freq: Minimum allowed output frequency, in Hz.
        melodia_trick : Whether to use the melodia trick to better detect notes.
        energy_tol: Drop notes below this energy.

    Returns:
        list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
        representing the note events, where amplitude is a number between 0 and 1
    """

    n_frames = frames.shape[0]

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        # add the note
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx + MIDI_OFFSET,
                amplitude,
            )
        )

    if melodia_trick:
        energy_shape = remaining_energy.shape

        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            # forward pass
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i += 1

            i_end = i - 1 - k  # go back to frame above threshold

            # backward pass
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i -= 1

            i_start = i + 1 + k  # go back to frame above threshold
            assert i_start >= 0, "{}".format(i_start)
            assert i_end < n_frames

            if i_end - i_start <= min_note_len:
                # note is too short, skip it
                continue

            # add the note
            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append(
                (
                    i_start,
                    i_end,
                    freq_idx + MIDI_OFFSET,
                    amplitude,
                )
            )

    return note_events





def log_base_b(x: Tensor, base: int) -> Tensor:
    """
    Compute log_b(x)
    Args:
        x : input
        base : log base. E.g. for log10 base=10
    Returns:
        log_base(x)
    """
    numerator = torch.log(x)
    denominator = torch.log(torch.tensor([base], dtype=numerator.dtype, device=numerator.device))
    return numerator / denominator


def normalized_log(inputs: Tensor) -> Tensor:
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """
    power = torch.square(inputs)
    log_power = 10 * log_base_b(power + 1e-10, 10)

    log_power_min = torch.amin(log_power, dim=(1, 2)).reshape(inputs.shape[0], 1, 1)
    log_power_offset = log_power - log_power_min    
    log_power_offset_max = torch.amax(log_power_offset, dim=(1, 2)).reshape(inputs.shape[0], 1, 1)
    # equivalent to TF div_no_nan
    log_power_normalized = log_power_offset / log_power_offset_max
    log_power_normalized = torch.nan_to_num(log_power_normalized, nan=0.0)

    return log_power_normalized.reshape(inputs.shape)


def get_cqt(
        inputs: Tensor, 
        n_harmonics: int, 
        use_batch_norm: bool, 
        bn_layer: nn.BatchNorm2d, 
    ):
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.
        use_batchnorm: If True, applies batch normalization after computing the CQT

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = np.min(
        [
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ]
    )
    cqt_layer = CQT2010v2(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        verbose=False,
    )
    cqt_layer.to(inputs.device)
    x = cqt_layer(inputs)
    x = torch.transpose(x, 1, 2)
    x = normalized_log(x)
    
    x = x.unsqueeze(1)
    if use_batch_norm:
        x = bn_layer(x)
    x = x.squeeze(1)
    
    return x


class HarmonicStacking(nn.Module):
    """Harmonic stacking layer

    Input shape: (n_batch, n_times, n_freqs, 1)
    Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

    n_freqs should be much larger than n_output_freqs so that information from the upper
    harmonics is captured.

    Attributes:
        bins_per_semitone: The number of bins per semitone of the input CQT
        harmonics: List of harmonics to use. Should be positive numbers.
        shifts: A list containing the number of bins to shift in frequency for each harmonic
        n_output_freqs: The number of frequency bins in each harmonic layer.
    """

    def __init__(
        self, 
        bins_per_semitone: int, 
        harmonics: List[float], 
        n_output_freqs: int,
    ):
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs

        self.shifts = [
            int(round(12.0 * self.bins_per_semitone * math.log2(h))) for h in self.harmonics
        ]
    
    @torch.no_grad()
    def forward(self, x):
        # x: (batch, t, n_bins)
        hcqt = []
        for shift in self.shifts:
            if shift == 0:
                cur_cqt = x
            if shift > 0:
                cur_cqt = F.pad(x[:, :, shift:], (0, shift))
            elif shift < 0:     # sub-harmonic
                cur_cqt = F.pad(x[:, :, :shift], (-shift, 0))
            hcqt.append(cur_cqt)
        hcqt = torch.stack(hcqt, dim=1)
        hcqt = hcqt[:, :, :, :self.n_output_freqs]
        return hcqt


class BasicPitchTorch(nn.Module):

    def __init__(
        self, 
        stack_harmonics=[0.5, 1, 2, 3, 4, 5, 6, 7],
    ) -> None:
        super().__init__()
        self.stack_harmonics = stack_harmonics
        if len(stack_harmonics) > 0:
            self.hs = HarmonicStacking(
                bins_per_semitone=CONTOURS_BINS_PER_SEMITONE, 
                harmonics=stack_harmonics, 
                n_output_freqs=ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
            )
            num_in_channels = len(stack_harmonics)
        else:
            num_in_channels = 1

        self.bn_layer = nn.BatchNorm2d(1, eps=0.001)
        self.conv_contour = nn.Sequential(
            # NOTE: in the original implementation, this part of the network should be dangling...
            # nn.Conv2d(num_in_channels, 32, kernel_size=5, padding="same"),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Conv2d(num_in_channels, 8, kernel_size=(3, 3 * 13), padding="same"),
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=5, padding="same"),
            nn.Sigmoid()
        )
        self.conv_note = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(7, 3), padding="same"),
            nn.Sigmoid()
        )
        self.conv_onset_pre = nn.Sequential(
            nn.Conv2d(num_in_channels, 32, kernel_size=5, stride=(1, 3)),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(),
        )
        self.conv_onset_post = nn.Sequential(
            nn.Conv2d(32 + 1, 1, kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cqt = get_cqt(
            x, 
            len(self.stack_harmonics), 
            True, 
            self.bn_layer,
        )
        if hasattr(self, "hs"):
            cqt = self.hs(cqt)
        else:
            cqt = cqt.unsqueeze(1)
                
        x_contour = self.conv_contour(cqt)

        # for strided conv, padding is different between PyTorch and TensorFlow
        # we use this equation: pad = [ (stride * (output-1)) - input + kernel ] / 2
        # (172, 264) --(1, 3)--> (172, 88), pad = ((1 * 171 - 172 + 7) / 2, (3 * 87 - 264 + 7) / 2) = (3, 2)
        # F.pad process from the last dimension, so it's (2, 2, 3, 3)
        x_contour_for_note = F.pad(x_contour, (2,2,3,3))
        x_note = self.conv_note(x_contour_for_note)
        
        # (172, 264) --(1, 3)--> (172, 88), pad = ((1 * 171 - 172 + 5) / 2, (3 * 87 - 264 + 5) / 2) = (2, 1)
        # F.pad process from the last dimension, so it's (1, 1, 2, 2)
        cqt_for_onset = F.pad(cqt, (1,1,2,2))
        x_onset_pre = self.conv_onset_pre(cqt_for_onset)
        x_onset_pre = torch.cat([x_note, x_onset_pre], dim=1)
        x_onset = self.conv_onset_post(x_onset_pre)
        outputs = {"onset": x_onset.squeeze(1), "contour": x_contour.squeeze(1), "note": x_note.squeeze(1)}
        return outputs


class BasicPitch():
    def __init__(self,model_path=hf_hub_download("shethjenil/Audio2Midi_Models","basicpitch/nmp.pth"),device="cpu"):
        self.model = BasicPitchTorch()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def run_inference(
        self,
        audio_path: str,
        progress_callback: Callable[[int, int], None] = None
    ) -> Dict[str, np.array]:
        audio_windowed, _, audio_original_length = get_audio_input(audio_path, OVERLAP_LEN, HOP_SIZE)
        audio_windowed = torch.from_numpy(np.copy(audio_windowed)).T.to(self.device)  # Shape: [num_windows, window_len]

        outputs = []
        total = audio_windowed.shape[0]

        with torch.no_grad():
            for i, window in enumerate(audio_windowed):
                window = window.unsqueeze(0)  # Add batch dimension
                output = self.model(window)
                outputs.append(output)

                # Call the callback if provided
                if progress_callback:
                    progress_callback(i + 1, total)

        # Merge outputs (assuming model returns a dict of tensors)
        merged_output = {}
        for key in outputs[0]:
            merged_output[key] = torch.cat([o[key] for o in outputs], dim=0)

        unwrapped_output = {
            k: unwrap_output(merged_output[k], audio_original_length, N_OVERLAPPING_FRAMES)
            for k in merged_output
        }
        return unwrapped_output

    def predict(self,audio,onset_thresh=0.5,frame_thresh=0.3,min_note_len=11,midi_tempo=120,infer_onsets=True,include_pitch_bends=True,multiple_pitch_bends=True,melodia_trick=True,progress_callback: Callable[[int, int], None] = None,min_freqat=None,max_freqat=None,output_file="output.mid"):
        int(np.round(min_note_len / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        model_output_to_notes(self.run_inference(audio,progress_callback),onset_thresh  = onset_thresh,frame_thresh  = frame_thresh,infer_onsets  = infer_onsets,min_note_len  = min_note_len,min_freq  = min_freqat,max_freq  = max_freqat,include_pitch_bends  = include_pitch_bends,multiple_pitch_bends  = multiple_pitch_bends,melodia_trick  = melodia_trick,midi_tempo  = midi_tempo).write(output_file)
        return output_file
