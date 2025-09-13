import numpy as np
from librosa.sequence import viterbi_discriminative
from librosa import note_to_hz,midi_to_hz , load as librosa_load
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt ,argrelmax
from torchaudio.models.conformer import ConformerLayer
from torch import cat as torch_cat , load as torch_load , from_numpy as torch_from_numpy,no_grad as torch_no_grad ,mean as torch_mean,std as torch_std,sigmoid as torch_sigmoid,nan_to_num as torch_nan_to_num,nn
from pretty_midi_fix import PrettyMIDI , Instrument , Note , PitchBend , instrument_name_to_program ,note_name_to_number
from typing import Callable, Dict, List, Optional, Tuple , Literal
from huggingface_hub import hf_hub_download

from mir_eval.melody import hz2cents




class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold,
                 offset_threshold, frame_threshold, pedal_offset_threshold,
                 begin_note):
        """Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
          frames_per_second: float
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = begin_note
        self.velocity_scale = 128

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num),
            'reg_offset_output': (segment_frames, classes_num),
            'frame_output': (segment_frames, classes_num),
            'velocity_output': (segment_frames, classes_num),
            'reg_pedal_onset_output': (segment_frames, 1),
            'reg_pedal_offset_output': (segment_frames, 1),
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83},
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96},
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """
        output_dict['frame_output'] = output_dict['note']
        output_dict['velocity_output'] = output_dict['note']
        output_dict['reg_onset_output'] = output_dict['onset']
        output_dict['reg_offset_output'] = output_dict['offset']
        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num),
            'reg_offset_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'velocity_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time,
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65],
             [11.98, 12.11, 33, 0.69],
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time
            and offset_time. E.g. [
             [0.17, 0.96],
             [1.17, 2.65],
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'],
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output  # Values are 0 or 1
        output_dict['onset_shift_output'] = onset_shift_output

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'],
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output  # Values are 0 or 1
        output_dict['offset_shift_output'] = offset_shift_output

        if 'reg_pedal_onset_output' in output_dict.keys():
            """Pedal onsets are not used in inference. Instead, frame-wise pedal
            predictions are used to detect onsets. We empirically found this is 
            more accurate to detect pedal onsets."""
            pass

        if 'reg_pedal_offset_output' in output_dict.keys():
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output) = \
                self.get_binarized_output_from_regression(
                    reg_output=output_dict['reg_pedal_offset_output'],
                    threshold=self.pedal_offset_threshold, neighbour=4)

            output_dict['pedal_offset_output'] = pedal_offset_output  # Values are 0 or 1
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        est_pedal_on_offs = None

        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape

        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets,
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """

        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]

        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = self.note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note],
                onset_output=output_dict['onset_output'][:, piano_note],
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note],
                offset_output=output_dict['offset_output'][:, piano_note],
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note],
                velocity_output=output_dict['velocity_output'][:, piano_note],
                frame_threshold=self.frame_threshold)

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)  # (notes, 5)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes)  # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]

        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times,
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]

        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0],
                'offset_time': est_on_off_note_vels[i][1],
                'midi_note': int(est_on_off_note_vels[i][2]),
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def note_detection_with_onset_offset_regress(self,frame_output, onset_output,
                                                    onset_shift_output, offset_output, offset_shift_output, velocity_output,
                                                    frame_threshold):
            """Process prediction matrices to note events information.
            First, detect onsets with onset outputs. Then, detect offsets
            with frame and offset outputs.

            Args:
            frame_output: (frames_num,)
            onset_output: (frames_num,)
            onset_shift_output: (frames_num,)
            offset_output: (frames_num,)
            offset_shift_output: (frames_num,)
            velocity_output: (frames_num,)
            frame_threshold: float
            Returns:
            output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
            e.g., [
                [1821, 1909, 0.47498, 0.3048533, 0.72119445],
                [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
                ...]
            """
            output_tuples = []
            bgn = None
            frame_disappear = None
            offset_occur = None

            for i in range(onset_output.shape[0]):
                if onset_output[i] == 1:
                    """Onset detected"""
                    if bgn:
                        """Consecutive onsets. E.g., pedal is not released, but two 
                        consecutive notes being played."""
                        fin = max(i - 1, 0)
                        output_tuples.append([bgn, fin, onset_shift_output[bgn],
                                            0, velocity_output[bgn]])
                        frame_disappear, offset_occur = None, None
                    bgn = i

                if bgn and i > bgn:
                    """If onset found, then search offset"""
                    if frame_output[i] <= frame_threshold and not frame_disappear:
                        """Frame disappear detected"""
                        frame_disappear = i

                    if offset_output[i] == 1 and not offset_occur:
                        """Offset detected"""
                        offset_occur = i

                    if frame_disappear:
                        if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                            """bgn --------- offset_occur --- frame_disappear"""
                            fin = offset_occur
                        else:
                            """bgn --- offset_occur --------- frame_disappear"""
                            fin = frame_disappear
                        output_tuples.append([bgn, fin, onset_shift_output[bgn],
                                            offset_shift_output[fin], velocity_output[bgn]])
                        bgn, frame_disappear, offset_occur = None, None, None

                    if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                        """Offset not detected"""
                        fin = i
                        output_tuples.append([bgn, fin, onset_shift_output[bgn],
                                            offset_shift_output[fin], velocity_output[bgn]])
                        bgn, frame_disappear, offset_occur = None, None, None

            # Sort pairs by onsets
            output_tuples.sort(key=lambda pair: pair[0])

            return output_tuples

class PerformanceLabel:
    """
    The dataset labeling class for performance representations. Currently, includes onset, note, and fine-grained f0
    representations. Note min, note max, and f0_bin_per_semitone values are to be arranged per instrument. The default
    values are for violin performance analysis. Fretted instruments might not require such f0 resolutions per semitone.
    """
    def __init__(self, note_min='F#3', note_max='C8', f0_bins_per_semitone=9, f0_smooth_std_c=None,
                 onset_smooth_std=0.7, f0_tolerance_c=200):
        midi_min = note_name_to_number(note_min)
        midi_max = note_name_to_number(note_max)
        self.midi_centers = np.arange(midi_min, midi_max)
        self.onset_smooth_std=onset_smooth_std # onset smoothing along time axis (compensate for alignment)

        f0_hz_range = note_to_hz([note_min, note_max])
        f0_c_min, f0_c_max = hz2cents(f0_hz_range)
        self.f0_granularity_c = 100/f0_bins_per_semitone
        if not f0_smooth_std_c:
            f0_smooth_std_c = self.f0_granularity_c * 5/4  # Keep the ratio from the CREPE paper (20 cents and 25 cents)
        self.f0_smooth_std_c = f0_smooth_std_c

        self.f0_centers_c = np.arange(f0_c_min, f0_c_max, self.f0_granularity_c)
        self.f0_centers_hz = 10 * 2 ** (self.f0_centers_c / 1200)
        self.f0_n_bins = len(self.f0_centers_c)

        self.pdf_normalizer = norm.pdf(0)

        self.f0_c2hz = lambda c: 10*2**(c/1200)
        self.f0_hz2c = hz2cents
        self.midi_centers_c = self.f0_hz2c(midi_to_hz(self.midi_centers))

        self.f0_tolerance_bins = int(f0_tolerance_c/self.f0_granularity_c)
        self.f0_transition_matrix = gaussian_filter1d(np.eye(2*self.f0_tolerance_bins + 1), 25/self.f0_granularity_c)

    def f0_c2label(self, pitch_c):
        """
        Convert a single f0 value in cents to a one-hot label vector with smoothing (i.e., create a gaussian blur around
        the target f0 bin for regularization and training stability. The blur is controlled by self.f0_smooth_std_c
        :param pitch_c: a single pitch value in cents
        :return: one-hot label vector with frequency blur
        """
        result = norm.pdf((self.f0_centers_c - pitch_c) / self.f0_smooth_std_c).astype(np.float32)
        result /= self.pdf_normalizer
        return result

    def f0_label2c(self, salience, center=None):
        """
        Convert the salience predictions to monophonic f0 in cents. Only outputs a single f0 value per frame!
        :param salience: f0 activations
        :param center: f0 center bin to calculate the weighted average. Use argmax if empty
        :return: f0 array per frame (in cents).
        """
        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(salience * self.f0_centers_c[start:end])
            weight_sum = np.sum(salience)
            return product_sum / np.clip(weight_sum, 1e-8, None)
        if salience.ndim == 2:
            return np.array([self.f0_label2c(salience[i, :]) for i in range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")

    def fill_onset_matrix(self, onsets, window, feature_rate):
        """
        Create a sparse onset matrix from window and onsets (per-semitone). Apply a gaussian smoothing (along time)
        so that we can tolerate better the alignment problems. This is similar to the frequency smoothing for the f0.
        The temporal smoothing is controlled by the parameter self.onset_smooth_std
        :param onsets: A 2d np.array of individual note onsets with their respective time values
        (Nx2: time in seconds - midi number)
        :param window: Timestamps for the frame centers of the sparse matrix
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds
        :return: onset_roll: A sparse matrix filled with temporally blurred onsets.
        """
        onsets = self.get_window_feats(onsets, window, feature_rate)
        onset_roll = np.zeros((len(window), len(self.midi_centers)))
        for onset in onsets:
            onset, note = onset  # it was a pair with time and midi note
            if self.midi_centers[0] < note < self.midi_centers[-1]: # midi note should be in the range defined
                note = int(note) - self.midi_centers[0]  # find the note index in our range
                onset = (onset*feature_rate)-window[0]    # onset index (as float but in frames, not in seconds!)
                start = max(0, int(onset) - 3)
                end = min(len(window) - 1, int(onset) + 3)
                try:
                    vals = norm.pdf(np.linspace(start - onset, end - onset, end - start + 1) / self.onset_smooth_std)
                    # if you increase 0.7 you smooth the peak
                    # if you decrease it, e.g., 0.1, it becomes too peaky! around 0.5-0.7 seems ok
                    vals /= self.pdf_normalizer
                    onset_roll[start:end + 1, note] += vals
                except ValueError:
                    print('start',start, 'onset', onset, 'end', end)
        return onset_roll, onsets

    def fill_note_matrix(self, notes, window, feature_rate):
        """
        Create the note matrix (piano roll) from window timestamps and note values per frame.
        :param notes: A 2d np.array of individual notes with their active time values Nx2
        :param window: Timestamps for the frame centers of the output
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds
        :return note_roll: The piano roll in the defined range of [note_min, note_max).
        """
        notes = self.get_window_feats(notes, window, feature_rate)

        # take the notes in the midi range defined
        notes = notes[np.logical_and(notes[:,1]>=self.midi_centers[0], notes[:,1]<=self.midi_centers[-1]),:]

        times = (notes[:,0]*feature_rate - window[0]).astype(int) # in feature samples (fs:self.hop/self.sr)
        notes = (notes[:,1] - self.midi_centers[0]).astype(int)

        note_roll = np.zeros((len(window), len(self.midi_centers)))
        note_roll[(times, notes)] = 1
        return note_roll, notes

    def fill_f0_matrix(self, f0s, window, feature_rate):
        """
        Unlike the labels for onsets and notes, f0 label is only relevant for strictly monophonic regions! Thus, this
        function returns a boolean which represents where to apply the given values.
        Never back-propagate without the boolean! Empty frames mean that the label is not that reliable.

        :param f0s: A 2d np.array of f0 values with the time they belong to (2xN: time in seconds - f0 in Hz)
        :param window: Timestamps for the frame centers of the output
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds

        :return f0_roll: f0 label matrix and
                f0_hz: f0 values in Hz
                annotation_bool: A boolean array representing which frames have reliable f0 annotations.
        """
        f0s = self.get_window_feats(f0s, window, feature_rate)
        f0_cents = np.zeros_like(window, dtype=float)
        f0s[:,1] = self.f0_hz2c(f0s[:,1]) # convert f0 in hz to cents

        annotation_bool = np.zeros_like(window, dtype=bool)
        f0_roll = np.zeros((len(window), len(self.f0_centers_c)))
        times_in_frame = f0s[:, 0]*feature_rate - window[0]
        for t, f0 in enumerate(f0s):
            t = times_in_frame[t]
            if t%1 < 0.25: # only consider it as annotation if the f0 values is really close to the frame center
                t = int(np.round(t))
                f0_roll[t] = self.f0_c2label(f0[1])
                annotation_bool[t] = True
                f0_cents[t] = f0[1]

        return f0_roll, f0_cents, annotation_bool

    @staticmethod
    def get_window_feats(time_feature_matrix, window, feature_rate):
        """
        Restrict the feature matrix to the features that are inside the window
        :param window: Timestamps for the frame centers of the output
        :param time_feature_matrix: A 2d array of Nx2 per the entire file.
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds
        :return: window_features: the features inside the given window
        """
        start = time_feature_matrix[:,0]>(window[0]-0.5)/feature_rate
        end = time_feature_matrix[:,0]<(window[-1]+0.5)/feature_rate
        window_features = np.logical_and(start, end)
        window_features = np.array(time_feature_matrix[window_features,:])
        return window_features

    def represent_midi(self, midi, feature_rate):
        """
        Represent a midi file as sparse matrices of onsets, offsets, and notes. No f0 is included.
        :param midi: A midi file (either a path or a pretty_midi_fix.PrettyMIDI object)
        :param feature_rate: The feature rate in Hz
        :return: dict {onset, offset, note, time}: Same format with the model's learning and outputs
        """
        def _get_onsets_offsets_frames(midi_content):
            if isinstance(midi_content, str):
                midi_content = PrettyMIDI(midi_content)
            onsets = []
            offsets = []
            frames = []
            for instrument in midi_content.instruments:
                for note in instrument.notes:
                    start = int(np.round(note.start * feature_rate))
                    end = int(np.round(note.end * feature_rate))
                    note_times = (np.arange(start, end+0.5)/feature_rate)[:, np.newaxis]
                    note_pitch = np.full_like(note_times, fill_value=note.pitch)
                    onsets.append([note.start, note.pitch])
                    offsets.append([note.end, note.pitch])
                    frames.append(np.hstack([note_times, note_pitch]))
            onsets = np.vstack(onsets)
            offsets = np.vstack(offsets)
            frames = np.vstack(frames)
            return onsets, offsets, frames, midi_content
        onset_array, offset_array, frame_array, midi_object = _get_onsets_offsets_frames(midi)
        window = np.arange(frame_array[0, 0]*feature_rate, frame_array[-1, 0]*feature_rate, dtype=int)
        onset_roll, _ = self.fill_onset_matrix(onset_array, window, feature_rate)
        offset_roll, _ = self.fill_onset_matrix(offset_array, window, feature_rate)
        note_roll, _ = self.fill_note_matrix(frame_array, window, feature_rate)
        start_anchor = onset_array[onset_array[:, 0]==np.min(onset_array[:, 0])]
        end_anchor = offset_array[offset_array[:, 0]==np.max(offset_array[:, 0])]
        return {
            'midi': midi_object,
            'note': note_roll,
            'onset': onset_roll,
            'offset': offset_roll,
            'time': window/feature_rate,
            'start_anchor': start_anchor,
            'end_anchor': end_anchor
        }

class NoPadConvBlock(nn.Module):
    def __init__(self, f, w, s, d, in_channels):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=(s, 1),
                                dilation=(d, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class TinyPathway(nn.Module):
    def __init__(self, dilation=1, hop=256, localize=False,
                 model_capacity="full", n_layers=6, chunk_size=256):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]
        self.layers = [1, 2, 3, 4, 5, 6]
        self.layers = self.layers[:n_layers]
        filters = [n * capacity_multiplier for n in [32, 8, 8, 8, 8, 8]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 32, 32]
        strides = self.deter_dilations(hop//(4*(2**n_layers)), localize=localize)
        strides[0] = strides[0]*4  # apply 4 times more stride at the first layer
        dilations = self.deter_dilations(dilation)

        for i in range(len(self.layers)):
            f, w, s, d, in_channel = filters[i + 1], widths[i], strides[i], dilations[i], filters[i]
            self.add_module("conv%d" % i, NoPadConvBlock(f, w, s, d, in_channel))
        self.chunk_size = chunk_size
        self.input_window, self.hop = self.find_input_size_for_pathway()
        self.out_dim = filters[n_layers]

    def find_input_size_for_pathway(self):
        def find_input_size(output_size, kernel_size, stride, dilation, padding):
            num = (stride*(output_size-1)) + 1
            input_size = num - 2*padding + dilation*(kernel_size-1)
            return input_size
        conv_calc, n = {}, 0
        for i in self.layers:
            layer = self.__getattr__("conv%d" % (i-1))
            for mm in layer.modules():
                if hasattr(mm, 'kernel_size'):
                    try:
                        d = mm.dilation[0]
                    except TypeError:
                        d = mm.dilation
                    conv_calc[n] = [mm.kernel_size[0], mm.stride[0], 0, d]
                    n += 1
        out = self.chunk_size
        hop = 1
        for n in sorted(conv_calc.keys())[::-1]:
            kernel_size_n, stride_n, padding_n, dilation_n = conv_calc[n]
            out = find_input_size(out, kernel_size_n, stride_n, dilation_n, padding_n)
            hop = hop*stride_n
        return out, hop

    def deter_dilations(self, total_dilation, localize=False):
        n_layers = len(self.layers)
        if localize:  # e.g., 32*1023 window and 3 layers -> [1, 1, 32]
            a = [total_dilation] + [1 for _ in range(n_layers-1)]
        else:  # e.g., 32*1023 window and 3 layers -> [4, 4, 2]
            total_dilation = int(np.log2(total_dilation))
            a = []
            for layer in range(n_layers):
                this_dilation = int(np.ceil(total_dilation/(n_layers-layer)))
                a.append(2**this_dilation)
                total_dilation = total_dilation - this_dilation
        return a[::-1]

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)
        x = x.permute(0, 3, 2, 1)
        return x




class Pitch_Det(nn.Module):
    def __init__(
            self,
            pathway_multiscale: int = 32,
            num_pathway_layers: int = 2,
            chunk_size: int = 256,
            hop_length: int = 256,
            encoder_dim: int = 256,
            sr: int = 44100,
            num_heads: int = 4,
            ffn_dim: int = 128,
            num_separator_layers: int = 16,
            num_representation_layers: int = 4,
            depthwise_conv_kernel_size: int = 31,
            dropout: float = 0.25,
            use_group_norm: bool = False,
            convolution_first: bool = False,
            labeling=PerformanceLabel(),
            wiring='tiktok',
            model_capacity="full"
    ):
        super().__init__()
        self.labeling = labeling
        self.sr = sr
        self.window_size = 1024
        self.hop_length = hop_length
        self.f0_bins_per_semitone = int(np.round(100/self.labeling.f0_granularity_c))

        self.main = TinyPathway(dilation=1, hop=hop_length, localize=True,
                                n_layers=num_pathway_layers, chunk_size=chunk_size,model_capacity=model_capacity)
        self.attendant = TinyPathway(dilation=pathway_multiscale, hop=hop_length, localize=False,
                                     n_layers=num_pathway_layers, chunk_size=chunk_size,model_capacity=model_capacity)
        assert self.main.hop == self.attendant.hop  # they should output with the same sample rate
        print('hop in samples:', self.main.hop)
        self.input_window = self.attendant.input_window

        self.encoder_dim = encoder_dim
        self.dropout = nn.Dropout(dropout)

        # merge two streams into a conformer input
        self.stream_merger = nn.Sequential(self.dropout,
                                           nn.Linear(self.main.out_dim + self.attendant.out_dim, self.encoder_dim))



        print('main stream window:', self.main.input_window,
              ', attendant stream window:', self.attendant.input_window,
              ', conformer input dim:', self.encoder_dim)

        center = ((chunk_size - 1) * self.main.hop)  # region labeled with pitch track
        main_overlap = self.main.input_window - center
        main_overlap = [int(np.floor(main_overlap / 2)), int(np.ceil(main_overlap / 2))]
        attendant_overlap = self.attendant.input_window - center
        attendant_overlap = [int(np.floor(attendant_overlap / 2)), int(np.ceil(attendant_overlap / 2))]
        print('main frame overlap:', main_overlap, ', attendant frame overlap:', attendant_overlap)
        main_crop_relative = [attendant_overlap[0] - main_overlap[0], main_overlap[1] - attendant_overlap[1]]
        print('crop for main pathway', main_crop_relative)
        print("Total sequence duration is", self.attendant.input_window, 'samples')
        print('Main stream receptive field for one frame is', (self.main.input_window - center), 'samples')
        print('Attendant stream receptive field for one frame is', (self.attendant.input_window - center), 'samples')
        self.frame_overlap = attendant_overlap

        self.main_stream_crop = main_crop_relative
        self.max_window_size = self.attendant.input_window
        self.chunk_size = chunk_size

        self.separator_stream = nn.ModuleList( # source-separation, reinvented
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_separator_layers)
            ]
        )

        self.f0_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.f0_head = nn.Linear(self.encoder_dim, len(self.labeling.f0_centers_c))

        self.note_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.note_head = nn.Linear(self.encoder_dim, len(self.labeling.midi_centers))

        self.onset_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.onset_head = nn.Linear(self.encoder_dim, len(self.labeling.midi_centers))

        self.offset_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.offset_head = nn.Linear(self.encoder_dim, len(self.labeling.midi_centers))

        self.labeling = labeling
        self.double_merger = nn.Sequential(self.dropout, nn.Linear(2 * self.encoder_dim, self.encoder_dim))
        self.triple_merger = nn.Sequential(self.dropout, nn.Linear(3 * self.encoder_dim, self.encoder_dim))
        self.wiring = wiring

        print('Total parameter count: ', self.count_parameters())

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def stream(self, x, representation, key_padding_mask=None):
        for i, layer in enumerate(self.__getattr__('{}_stream'.format(representation))):
            x = layer(x, key_padding_mask)
        return x

    def head(self, x, representation):
        return self.__getattr__('{}_head'.format(representation))(x)

    def forward(self, x, key_padding_mask=None):

        # two auditory streams followed by the separator stream to ensure timbre-awareness
        x_attendant = self.attendant(x)
        x_main = self.main(x[:, self.main_stream_crop[0]:self.main_stream_crop[1]])
        x = self.stream_merger(torch_cat((x_attendant, x_main), -1).squeeze(1))
        x = self.stream(x, 'separator', key_padding_mask)

        f0 = self.stream(x, 'f0', key_padding_mask) # they say this is a low level feature :)

        if self.wiring == 'parallel':
            note = self.stream(x, 'note', key_padding_mask)
            onset = self.stream(x, 'onset', key_padding_mask)
            offset = self.stream(x, 'offset', key_padding_mask)

        elif self.wiring == 'tiktok':
            onset = self.stream(x, 'onset', key_padding_mask)
            offset = self.stream(x, 'offset', key_padding_mask)
            # f0 is disconnected, note relies on separator, onset, and offset
            note = self.stream(self.triple_merger(torch_cat((x, onset, offset), -1)), 'note', key_padding_mask)

        elif self.wiring == 'tiktok2':
            onset = self.stream(x, 'onset', key_padding_mask)
            offset = self.stream(x, 'offset', key_padding_mask)
            # note is connected to f0, onset, and offset
            note = self.stream(self.triple_merger(torch_cat((f0, onset, offset), -1)), 'note', key_padding_mask)

        elif self.wiring == 'spotify':
            # note is connected to f0 only
            note = self.stream(f0, 'note', key_padding_mask)
            # here onset and onsets are higher-level features informed by the separator and note
            onset = self.stream(self.double_merger(torch_cat((x, note), -1)), 'onset', key_padding_mask)
            offset = self.stream(self.double_merger(torch_cat((x, note), -1)), 'offset', key_padding_mask)

        else:
            # onset and offset are connected to f0 and separator streams
            onset = self.stream(self.double_merger(torch_cat((x, f0), -1)), 'onset', key_padding_mask)
            offset = self.stream(self.double_merger(torch_cat((x, f0), -1)), 'offset', key_padding_mask)
            # note is connected to f0, onset, and offset streams
            note = self.stream(self.triple_merger(torch_cat((f0, onset, offset), -1)), 'note', key_padding_mask)


        return {'f0': self.head(f0, 'f0'),
                'note': self.head(note, 'note'),
                'onset': self.head(onset, 'onset'),
                'offset': self.head(offset, 'offset')}


class Violin_Pitch_Det(Pitch_Det):
    def __init__(self,device="cpu",model_path=None):
        if not model_path:
            model_path = hf_hub_download("shethjenil/Audio2Midi_Models","violin.pt")
        model_conf = {
        "wiring": "parallel",
        "sampling_rate": 44100,
        "pathway_multiscale": 4,
        "num_pathway_layers": 2,
        "num_separator_layers": 16,
        "num_representation_layers": 4,
        "hop_length": 256,
        "chunk_size": 512,
        "minSNR": -32,
        "maxSNR": 96,
        "note_low": "F#3",
        "note_high": "E8",
        "f0_bins_per_semitone": 10,
        "f0_smooth_std_c": 12,
        "onset_smooth_std": 0.7
        }
        super().__init__(pathway_multiscale=model_conf['pathway_multiscale'],num_pathway_layers=model_conf['num_pathway_layers'], wiring=model_conf['wiring'],hop_length=model_conf['hop_length'], chunk_size=model_conf['chunk_size'],labeling=PerformanceLabel(note_min=model_conf['note_low'], note_max=model_conf['note_high'],f0_bins_per_semitone=model_conf['f0_bins_per_semitone'],f0_tolerance_c=200,f0_smooth_std_c=model_conf['f0_smooth_std_c'], onset_smooth_std=model_conf['onset_smooth_std']), sr=model_conf['sampling_rate'],model_capacity="full")
        self.load_state_dict(torch_load(model_path, map_location=device,weights_only=True))
        self.eval()

    def out2note(self, output: Dict[str, np.array], postprocessing='spotify',
                 include_pitch_bends: bool = True,
    ) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
        """Convert model output to notes
        """
        if postprocessing == 'spotify':
            estimated_notes = self.spotify_create_notes(
                output["note"],
                output["onset"],
                note_low=self.labeling.midi_centers[0],
                note_high=self.labeling.midi_centers[-1],
                onset_thresh=0.5,
                frame_thresh=0.3,
                infer_onsets=True,
                min_note_len=int(np.round(127.70 / 1000 * (self.sr / self.hop_length))), #127.70
                melodia_trick=True,
            )

        elif postprocessing == 'tiktok':
            postprocessor = RegressionPostProcessor(
                frames_per_second=self.sr / self.hop_length,
                classes_num=self.labeling.midi_centers.shape[0],
                begin_note=self.labeling.midi_centers[0],
                onset_threshold=0.2,
                offset_threshold=0.2,
                frame_threshold=0.3,
                pedal_offset_threshold=0.5,
            )
            tiktok_note_dict, _ = postprocessor.output_dict_to_midi_events(output)
            estimated_notes = []
            for list_item in tiktok_note_dict:
                if list_item['offset_time'] > 0.6 + list_item['onset_time']:
                    estimated_notes.append((int(np.floor(list_item['onset_time']/(output['time'][1]))),
                                            int(np.ceil(list_item['offset_time']/(output['time'][1]))),
                                            list_item['midi_note'], list_item['velocity']/128))

        if include_pitch_bends:
            estimated_notes_with_pitch_bend = self.get_pitch_bends(output["f0"], estimated_notes)
        else:
            estimated_notes_with_pitch_bend = [(note[0], note[1], note[2], note[3], None) for note in estimated_notes]

        times_s = output['time']
        estimated_notes_time_seconds = [
            (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
        ]

        return estimated_notes_time_seconds

    def note2midi(
        self,
        note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
        midi_tempo: float = 120,
    ):
        """Create a pretty_midi_fix object from note events
            :param note_events_with_pitch_bends: list of tuples
                    [(start_time_seconds, end_time_seconds, pitch_midi, amplitude, [pitch_bend])]
            :param midi_tempo: MIDI tempo (BPM)
            :return: PrettyMIDI object
        """
        mid = PrettyMIDI(initial_tempo=midi_tempo)
        
        # Create a single instrument (e.g., program=40 = violin)
        instrument = Instrument(program=40)
        
        for start_time, end_time, note_number, amplitude, pitch_bend in note_events_with_pitch_bends:
            note = Note(
                velocity=int(np.round(127 * amplitude)),
                pitch=note_number,
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(note)

            if pitch_bend is not None and isinstance(pitch_bend, (list, np.ndarray)):
                pitch_bend = np.asarray(pitch_bend)
                pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))
                for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend):
                    instrument.pitch_bends.append(PitchBend(pb_midi, pb_time))

        # Add the single instrument to the MIDI object
        mid.instruments.append(instrument)

        return mid

    def get_pitch_bends(
            self,
            contours: np.ndarray, note_events: List[Tuple[int, int, int, float]],
            timing_refinement_range: int = 0, to_midi: bool = True,
    ) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
        """
        Given note events and contours, estimate pitch bends per note.
        Pitch bends are represented as a sequence of evenly spaced midi pitch bend control units.
        The time stamps of each pitch bend can be inferred by computing an evenly spaced grid between
        the start and end times of each note event.
        Args:
            contours: Matrix of estimated pitch contours
            note_events: note event tuple
            timing_refinement_range: if > 0, refine onset/offset boundaries with f0 confidence
            to_midi: whether to convert pitch bends to midi pitch bends. If False, return pitch estimates in the format
        [time (index), pitch (Hz), confidence in range [0, 1]].
        Returns:
            note events with pitch bends
        """

        f0_matrix = []  # [time (index), pitch (Hz), confidence in range [0, 1]]
        note_events_with_pitch_bends = []
        for start_idx, end_idx, pitch_midi, amplitude in note_events:
            if timing_refinement_range:
                start_idx = np.max([0, start_idx - timing_refinement_range])
                end_idx = np.min([contours.shape[0], end_idx + timing_refinement_range])
            freq_idx = int(np.round(self.midi_pitch_to_contour_bin(pitch_midi)))
            freq_start_idx = np.max([freq_idx - self.labeling.f0_tolerance_bins, 0])
            freq_end_idx = np.min([self.labeling.f0_n_bins, freq_idx + self.labeling.f0_tolerance_bins + 1])

            trans_start_idx = np.max([0, self.labeling.f0_tolerance_bins - freq_idx])
            trans_end_idx = (2 * self.labeling.f0_tolerance_bins + 1) - \
                            np.max([0, freq_idx - (self.labeling.f0_n_bins - self.labeling.f0_tolerance_bins - 1)])

            # apply regional viterbi to estimate the intonation
            # observation probabilities come from the f0_roll matrix
            observation = contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            observation = observation / observation.sum(axis=1)[:, None]
            observation[np.isnan(observation.sum(axis=1)), :] = np.ones(freq_end_idx - freq_start_idx) * 1 / (
                        freq_end_idx - freq_start_idx)

            # transition probabilities assure continuity
            transition = self.labeling.f0_transition_matrix[trans_start_idx:trans_end_idx,
                         trans_start_idx:trans_end_idx] + 1e-6
            transition = transition / np.sum(transition, axis=1)[:, None]

            path = viterbi_discriminative(observation.T / observation.sum(axis=1), transition) + freq_start_idx

            cents = np.array([self.labeling.f0_label2c(contours[i + start_idx, :], path[i]) for i in range(len(path))])
            bends = cents - self.labeling.midi_centers_c[pitch_midi - self.labeling.midi_centers[0]]
            if to_midi:
                bends = (bends * 4096 / 100).astype(int)
                bends[bends > 8191] = 8191
                bends[bends < -8192] = -8192

                if timing_refinement_range:
                    confidences = np.array([contours[i + start_idx, path[i]] for i in range(len(path))])
                    threshold = np.median(confidences)
                    threshold = (np.median(confidences > threshold) + threshold) / 2  # some magic
                    median_kernel = 2 * (timing_refinement_range // 2) + 1  # some more magic
                    confidences = medfilt(confidences, kernel_size=median_kernel)
                    conf_bool = confidences > threshold
                    onset_idx = np.argmax(conf_bool)
                    offset_idx = len(confidences) - np.argmax(conf_bool[::-1])
                    bends = bends[onset_idx:offset_idx]
                    start_idx = start_idx + onset_idx
                    end_idx = start_idx + offset_idx

                note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
            else:
                confidences = np.array([contours[i + start_idx, path[i]] for i in range(len(path))])
                time_idx = np.arange(len(path)) + start_idx
                # f0_hz = self.labeling.f0_c2hz(cents)
                possible_f0s = np.array([time_idx, cents, confidences]).T
                f0_matrix.append(possible_f0s[np.abs(bends)<100]) # filter out pitch bends that are too large
        if not to_midi:
            return np.vstack(f0_matrix)
        else:
            return note_events_with_pitch_bends

    def midi_pitch_to_contour_bin(self, pitch_midi: int) -> np.array:
        """Convert midi pitch to corresponding index in contour matrix
        Args:
            pitch_midi: pitch in midi
        Returns:
            index in contour matrix
        """
        pitch_hz = midi_to_hz(pitch_midi)
        return np.argmin(np.abs(self.labeling.f0_centers_hz - pitch_hz))

    def get_inferred_onsets(self,onset_roll: np.array, note_roll: np.array, n_diff: int = 2) -> np.array:
        """
        Infer onsets from large changes in note roll matrix amplitudes.
        Modified from https://github.com/spotify/basic-pitch/blob/main/basic_pitch/note_creation.py
        :param onset_roll: Onset activation matrix (n_times, n_freqs).
        :param note_roll: Frame-level note activation matrix (n_times, n_freqs).
        :param n_diff: Differences used to detect onsets.
        :return: The maximum between the predicted onsets and its differences.
        """

        diffs = []
        for n in range(1, n_diff + 1):
            frames_appended = np.concatenate([np.zeros((n, note_roll.shape[1])), note_roll])
            diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
        frame_diff = np.min(diffs, axis=0)
        frame_diff[frame_diff < 0] = 0
        frame_diff[:n_diff, :] = 0
        frame_diff = np.max(onset_roll) * frame_diff / np.max(frame_diff)  # rescale to have the same max as onsets

        max_onsets_diff = np.max([onset_roll, frame_diff],
                                axis=0)  # use the max of the predicted onsets and the differences

        return max_onsets_diff

    def spotify_create_notes(
            self,
            note_roll: np.array,
            onset_roll: np.array,
            onset_thresh: float,
            frame_thresh: float,
            min_note_len: int,
            infer_onsets: bool,
            note_low : int, #self.labeling.midi_centers[0]
            note_high : int, #self.labeling.midi_centers[-1],
            melodia_trick: bool = True,
            energy_tol: int = 11,
    ) -> List[Tuple[int, int, int, float]]:
        """Decode raw model output to polyphonic note events
        Modified from https://github.com/spotify/basic-pitch/blob/main/basic_pitch/note_creation.py
        Args:
            note_roll: Frame activation matrix (n_times, n_freqs).
            onset_roll: Onset activation matrix (n_times, n_freqs).
            onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
            frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
            min_note_len: Minimum allowed note length in frames.
            infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
            melodia_trick : Whether to use the melodia trick to better detect notes.
            energy_tol: Drop notes below this energy.
        Returns:
            list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
            representing the note events, where amplitude is a number between 0 and 1
        """

        n_frames = note_roll.shape[0]

        # use onsets inferred from frames in addition to the predicted onsets
        if infer_onsets:
            onset_roll = self.get_inferred_onsets(onset_roll, note_roll)

        peak_thresh_mat = np.zeros(onset_roll.shape)
        peaks = argrelmax(onset_roll, axis=0)
        peak_thresh_mat[peaks] = onset_roll[peaks]

        onset_idx = np.where(peak_thresh_mat >= onset_thresh)
        onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
        onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

        remaining_energy = np.zeros(note_roll.shape)
        remaining_energy[:, :] = note_roll[:, :]

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
            if freq_idx < note_high:
                remaining_energy[note_start_idx:i, freq_idx + 1] = 0
            if freq_idx > note_low:
                remaining_energy[note_start_idx:i, freq_idx - 1] = 0

            # add the note
            amplitude = np.mean(note_roll[note_start_idx:i, freq_idx])
            note_events.append(
                (
                    note_start_idx,
                    i,
                    freq_idx + note_low,
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
                    if freq_idx < note_high:
                        remaining_energy[i, freq_idx + 1] = 0
                    if freq_idx > note_low:
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
                    if freq_idx < note_high:
                        remaining_energy[i, freq_idx + 1] = 0
                    if freq_idx > note_low:
                        remaining_energy[i, freq_idx - 1] = 0

                    i -= 1

                i_start = i + 1 + k  # go back to frame above threshold
                assert i_start >= 0, "{}".format(i_start)
                assert i_end < n_frames

                if i_end - i_start <= min_note_len:
                    # note is too short, skip it
                    continue

                # add the note
                amplitude = np.mean(note_roll[i_start:i_end, freq_idx])
                note_events.append(
                    (
                        i_start,
                        i_end,
                        freq_idx + note_low,
                        amplitude,
                    )
                )

        return note_events
    
    def read_audio(self, audio):
        """
        Read and resample an audio file, convert to mono, and unfold into representation frames.
        The time array represents the center of each small frame with 5.8ms hop length. This is different than the chunk
        level frames. The chunk level frames represent the entire sequence the model sees. Whereas it predicts with the
        small frames intervals (5.8ms).
        :param  audio: str, pathlib.Path
        :return: frames: (n_big_frames, frame_length), times: (n_small_frames,)
        """
        audio = torch_from_numpy(librosa_load(audio, sr=self.sr, mono=True)[0])
        len_audio = audio.shape[-1]
        n_frames = int(np.ceil((len_audio + sum(self.frame_overlap)) / (self.hop_length * self.chunk_size)))
        audio = nn.functional.pad(audio, (self.frame_overlap[0],self.frame_overlap[1] + (n_frames * self.hop_length * self.chunk_size) - len_audio))
        frames = audio.unfold(0, self.max_window_size, self.hop_length*self.chunk_size)
        times = np.arange(0, len_audio, self.hop_length) / self.sr    # not tensor, we don't compute anything with it
        return frames, times

    def model_predict(self, audio, batch_size,progress_callback: Callable[[int, int], None]):
        device = self.main.conv0.conv2d.weight.device
        performance = {'f0': [], 'note': [], 'onset': [], 'offset': []}
        frames, times = self.read_audio(audio)
        with torch_no_grad():
            for i in range(0, len(frames), batch_size):
                f = frames[i:min(i + batch_size, len(frames))].to(device)
                f -= (torch_mean(f, axis=1).unsqueeze(-1))
                f /= (torch_std(f, axis=1).unsqueeze(-1))
                out = self.forward(f)
                for key, value in out.items():
                    value = torch_sigmoid(value)
                    value = torch_nan_to_num(value) # the model outputs nan when the frame is silent (this is an expected behavior due to normalization)
                    value = value.view(-1, value.shape[-1])
                    value = value.detach().cpu().numpy()
                    performance[key].append(value)
                if progress_callback:
                    progress_callback(i,len(frames))
        performance = {key: np.concatenate(value, axis=0)[:len(times)] for key, value in performance.items()}
        performance['time'] = times
        return performance

    def predict(self, audio, batch_size=32, postprocessing="spotify",include_pitch_bends=True,progress_callback: Callable[[int, int], None] = None,output_file="output.mid"):
        output = self.model_predict(audio, batch_size,progress_callback)
        self.note2midi(self.out2note(output, postprocessing, include_pitch_bends), 120).write(output_file)
        return output_file
